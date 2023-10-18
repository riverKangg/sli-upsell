
# Model Development Pipeline

이 프로젝트는 모델 개발을 위한 파이썬 코드와 모듈로 구성된 파이프라인이다. 아래에서 프로젝트를 실행하고 각 모듈을 설명한다.

## 설치

먼저 프로젝트를 클론한다.

```
git clone https://github.com/riverKangg/sli-upsell.git
```

그런 다음 프로젝트 디렉토리로 이동한다.

```
cd sli-upsell
```

다음으로 필요한 라이브러리 및 종속성을 설치한다.

```
pip install -r requirements.txt
```

## 데이터 준비

프로젝트를 실행하기 전에 학습 및 테스트 데이터를 프로젝트 디렉토리의 `data` 폴더에 추가해야 한다.

`sample_data_generator.py` 모듈로 샘플 데이터 생성 및 확인 가능

데이터 저장 규칙
- csv 형식 사용
- 파일명 끝은 마감년월(yyyymm형식)으로 저장
- 학습용 데이터는 타겟(PERF) 포함, 스코어 데이터는 타겟 미포함

## 모델 개발

프로젝트의 핵심은 `model_development_pipeline.py` 파일이다. 이 파일에서 모든 모델 개발 및 평가 작업이 수행된다.

```
python model_development_pipeline.py
```

이 명령을 실행하면 다음 단계가 수행된다:

1. 데이터 유효성 검사(Data Validation)
2. 데이터 전처리(Data Preprocessing)
3. 모델 학습(Model Training)
4. 모델 평가(Model Evaluation)

모든 중간 및 최종 결과물은 `output` 폴더에 저장된다.

## 모듈 설명

프로젝트의 각 모듈에 대한 간략한 설명은 다음과 같다:

- `data_validator.py`: 데이터 유효성 검사를 수행하는 모듈로, 데이터의 누락, 중복, NULL 값 등을 확인한다.
- `data_preprocessor.py`: 데이터 전처리를 수행하는 모듈로, 데이터 정리, 원-핫 인코딩, 결측치 처리 등을 수행한다.
- `model_trainer.py`: 모델 학습을 수행하는 모듈로, XGBoost 또는 LightGBM과 같은 모델을 학습한다.
- `model_evaluator.py`: 모델 평가를 수행하는 모듈로, ROC AUC, PSI 등의 모델 성능 메트릭을 계산한다.
- `score_calculator.py` : 최종 모델을 사용하여 스코어를 산출하고, 등급화하여 저장한다.
- `variable_iv_psi_calculator.py` : 변수를 구간화하고, IV와 PSI값을 산출하고 저장한다.

## Utils
- `utils/paths.py`: 데이터 및 모델 경로를 정의하는 모듈.
- `utils/keys.py`: 데이터 프레임의 열 키를 정의하는 모듈.
- `utils/hyperparameters.py`: 베이지안 최적화에 사용할 파라미터를 정의하는 모듈.

## 결과 확인

모든 중간 및 최종 결과물은 `output` 폴더에 저장됩니다. 학습된 모델, 모델 평가 결과 및 중간 데이터를 확인할 수 있다.

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.
