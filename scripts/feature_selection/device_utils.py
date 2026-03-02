"""GPU/CPU device handling with fallback."""
import logging

logger = logging.getLogger(__name__)


def fit_lgb_with_fallback(model, X, y, device: str):
    """
    model.fit(X, y) 실행. GPU 실패 시 CPU로 새 모델 생성 후 fit.
    Returns: fitted model (같은 객체 또는 CPU fallback 새 객체)
    """
    try:
        model.fit(X, y)
        return model
    except Exception as e:
        err_msg = str(e).lower()
        if device == "gpu" and (
            "opencl" in err_msg or "cuda" in err_msg or "device" in err_msg or "gpu" in err_msg
        ):
            logger.warning("LightGBM GPU 실패, CPU fallback: %s", str(e).split("\n")[0][:50])
            import lightgbm as lgb
            params = model.get_params()
            params["device"] = "cpu"
            m = lgb.LGBMRegressor(**params)
            m.fit(X, y)
            return m
        raise
