# Research Agent

LLM의 연구 아이디어 제안·비판·개선을 실험 설계와 코드 작성으로 연결하는 연구 자동화 프로젝트입니다. **시간창 제약 외판원 문제(TSPTW)**를 사례로, 역할별 프롬프트와 연구 토론 기록, 의사코드, 실험 코드를 정리했습니다.

## 연구 흐름

`문제 정의 → 아이디어 생성·문헌 검토 → 비판·개선 → 실험 방향 결정 → 의사코드·초안 작성 → 코드 구현·실험 → 결과 피드백`

- **역할 분리:** 아이디어 생성, 비판, 개선, 의사결정을 별도 프롬프트로 구성합니다.
- **문맥 관리:** 문제의 제약, 수식, 변경 이유를 기록해 후속 단계에 전달합니다.
- **실행 추적:** 웹 LLM 호출용 Python CLI에서 요청·응답·모델·종료 코드를 로그로 남깁니다.

## 주요 자료

| 경로 | 내용 |
| --- | --- |
| [연구 에이전트 명세서](공유용문서/research_agent_specification.pdf) | 토론과 실험을 반복하는 워크플로우 설계 |
| [research agent prompt/](research%20agent%20prompt/) | 초기 구현 요구사항과 역할별 프롬프트 |
| [웹 LLM 호출 가이드](공유용문서/web_llm_call/web_llm_call.md) | 브라우저 연동 방법, CLI·프롬프트·호출 로그 ZIP |
| [maybe_working/](maybe_working/) | TSPTW 연구 토론, LaTeX 초안·의사코드, Python/PyTorch 실험 코드 |
| [acadmm_archive/](acadmm_archive/) | 이전 ACADMM 기반 실험 코드 |

구체적인 예시는 [토론 기록](maybe_working/20260406_102057_debate.md) → [의사코드](maybe_working/20260406_142852_pseudocode.tex) → [실험 코드](maybe_working/agfn_tsptw.py) 순서로 볼 수 있습니다.

## 실행 범위

이 저장소는 설계와 실험 산출물을 공유하는 프로토타입 아카이브입니다. 전체 실행에는 별도의 supervisor·브라우저 호출 모듈, 로그인된 LLM 세션, 실험 데이터와 의존성이 필요합니다. 실행 환경은 [웹 LLM 호출 가이드](공유용문서/web_llm_call/web_llm_call.md)를 참고하세요.
