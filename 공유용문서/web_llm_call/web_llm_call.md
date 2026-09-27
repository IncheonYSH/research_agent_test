# Web Browser LLM Call Guide

이 문서는 "웹브라우저에서 로그인된 LLM"을 어떻게 호출하는지, 어떤 레이어를 거쳐 동작하는지, 그리고 실제 셋업과 실행은 어떻게 하는지를 정리한 운영 문서임.

## 1. 요약

웹 LLM 호출은 

* 매우 비싼 API 호출을 우회
* 코딩 이외의 task를 수행
* 보다 범용적인 목적으로 설계된 적절한 모델을 호출하고 그 응답을 다음 작업을 위한 컨텍스트로 제공

하기 위한 것임.

로그인된 브라우저 세션에 Playwright(브라우저 자동화 tool)가 CDP로 붙어서 ChatGPT Web 또는 Gemini Web에 직접 프롬프트를 넣고 응답을 회수하는 구조임.

Codex, CLAUDE code등의 코딩에이전트의 기본적인 추론 능력을 고려할 때, 웹 LLM 을 별도로 reasoning backbone으로 사용하는 것이 유의미한 이득이 있으려면 최고급 모델을 사용하는 것이 적절함함. 따라서 다음 두 종류의 서비스에 대해서 구현함.

- `ChatGPT Web`의 `GPT-5.4 Pro` + `Extended Thinking`
- `Gemini Web`의 `Gemini Pro 3.1`

둘 다 "공유 브라우저 세션"을 사용하지만, 각 요청은 새 채팅 또는 새 페이지로 격리해서 처리하고, 응답을 회수한 뒤 페이지를 닫는 방식으로 운용된다.

브라우저 자동화 도구를 사용하는 만큼, 에이전트가 브라우저를 조작 할 때, 작업용 pc에서 해당 스크립트를 구동하면 수동으로 다른 작업을 처리하지 못할 수 있으므로 **docker**환경에서의 구동을 추천함.

## 2. 구성 요소

1. SKILL 지정을 위한 cript
2. 수동 CLI 지정을 위한 script
   - 외부 harness, orchestrator 에서 호출하는 형태

- SKILL은 별도 harness, orchestrator(rule 기반의 workflow 강제 시스템)을 사용하지 않을 때 agent자체에서 반복 작업을 묶어 지정하여 사용하게 하는 기능임. 이 경우, `웹 브라우저 호출 자동화 스크립트 사용` 을 스킬로 지정하고(자세한 방법은 codex, claude code의 공식 문서 참고), 해당 스킬의 설명을 적절히 지정하면 됨.
- SKILL로 지정하고 사용하게 하면, gpt pro 등 응답 시간이 매우 긴 모델의 응답 시간을 끝까지 기다리지 못하는 현상이 있는데, 프롬프트 헤더에 `We have plenty of time and CPU, Memory, so don't worry about limits` 를 추가하면 완화 가능. 하지만 해당 프롬프트 만으로 문제를 완전히 해결할 수는 없음. pro 모델을 적극적으로 활용하려면 codex, web gpt pro 를 적절히 호출해서 관리하는 외부 orchestrator 가 필요함.

## 3. Workflow

### 3.1 GPT Pro

수동 CLI는 아래 조건을 강제한다.

- 모델: `GPT-5.4 Pro`
- Thinking mode: `Extended Thinking`
- 새 채팅에서 시작
- 완료 후 해당 페이지 닫기
- 모델 또는 thinking mode를 못 찾으면 실패

### 3.2 Gemini

Gemini 쪽은 아래를 강제한다.

- 모델: `Gemini Pro 3.1`
- 새 채팅에서 시작
- 완료 후 페이지 닫기
- 로그인되어 있어야 함
- 모델을 못 찾으면 실패

## 4. 실제 실행 흐름

1. 공유 브라우저 GUI 시작
2. 로그인 또는 readiness 검사
3. 필요하면 원격 접속용 tunnel URL 생성(**docker 사용시**)
4. 사용자가 브라우저에서 직접 로그인
5. 리포지토리 의존 작업이면 CTX handoff 문서 준비
6. 프롬프트 전송
7. JSON 응답 회수
8. 호출 로그 저장

### 4.1 GUI 시작(docker 사용시)

Linux 계열에서는 `/workspace/scripts/gpt_web_login/start-chatgpt-remote.sh`가 다음을 띄운다.

- `Xvfb`
- `openbox`
- `x11vnc`
- `websockify`
- `noVNC`
- `chromium` 또는 `google-chrome` 또는 Playwright Chromium

그리고 브라우저는 아래 조건으로 올라간다.

- `--remote-debugging-address=127.0.0.1`
- `--remote-debugging-port=9222`
- `--user-data-dir=/workspace/scripts/.chatgpt-profile`

즉 Playwright는 나중에 `http://127.0.0.1:9222`로 연결해 기존 persistent browser context를 재사용한다

### 4.3 로그인 대기

오케스트레이터나 수동 흐름에서 readiness가 실패하면 shared GUI와 tunnel을 띄우고, 아래 정보를 사용자에게 보여준다.

- `local_gui_url`
- `tunnel_url`
- `vnc_password`
- `cdp_endpoint`

사용자가 원격 브라우저에서 직접 로그인한 뒤 터미널로 돌아와 재시도하는 구조다.

주의:

- 이 흐름은 interactive terminal을 전제로 한다.
- stdin이 interactive하지 않으면 오케스트레이터는 로그인 대기 중 실패 상태로 끝난다.

## 5. Context Engineering

웹 LLM은 로컬 파일시스템을 직접 읽지 못함. 그래서 리포지토리 의존 질문은 반드시 "CTX handoff 문서"를 만들어 함께 보내는 방식으로 설계해야 함.

핵심 원칙은 다음이다.

- 파일 경로만 주면 안 된다.
- 함수명이나 line number만 줘도 안 된다.
- 실제 코드/로그/설정/관찰 결과를 사람이 읽을 수 있는 문장으로 요약해서 넣어야 한다.
