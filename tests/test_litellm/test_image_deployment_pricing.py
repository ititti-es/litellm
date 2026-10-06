"""Config-priced image models must bill from their deployment's per-image price.

The router stores custom pricing only under the deployment id, so the image
cost path has to read it from there instead of the provider/size cost map.
"""

import litellm
from litellm import Router
from litellm.types.utils import ImageObject, ImageResponse


def _router(deployment_id: str, model_info: dict) -> tuple[Router, str]:
    router = Router(
        model_list=[
            {
                "model_name": "qwen-image-3",
                "litellm_params": {
                    "model": "openai/qwen/qwen-image-3",
                    "api_base": "https://openrouter.ai/api/v1",
                    "api_key": "test-key",
                },
                "model_info": {"id": deployment_id, "mode": "image_generation", **model_info},
            }
        ]
    )
    return router, router.model_list[0]["model_info"]["id"]


def _image_response(count: int) -> ImageResponse:
    return ImageResponse(created=1, data=[ImageObject(b64_json="aGk=") for _ in range(count)])


def _cost(router_model_id: str, count: int) -> float:
    return litellm.completion_cost(
        completion_response=_image_response(count),
        model="openai/qwen/qwen-image-3",
        call_type="aimage_generation",
        custom_llm_provider="openai",
        size="1024x1024",
        custom_pricing=True,
        router_model_id=router_model_id,
    )


def test_config_priced_image_model_bills_per_image():
    _, model_id = _router("qwen-image-input-price", {"input_cost_per_image": 0.03})

    assert _cost(model_id, 1) == 0.03
    assert abs(_cost(model_id, 3) - 0.09) < 1e-12


def test_output_cost_per_image_is_also_honored():
    _, model_id = _router("qwen-image-output-price", {"output_cost_per_image": 0.04})

    assert _cost(model_id, 2) == 0.08


def test_deployment_without_image_price_keeps_existing_calculator():
    _, model_id = _router("qwen-image-unpriced", {})

    try:
        cost = _cost(model_id, 1)
    except Exception as e:
        assert "pricing" in str(e).lower() or "cost map" in str(e).lower()
    else:
        assert cost == 0.0
