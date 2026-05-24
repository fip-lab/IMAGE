import requests

SERVER_URL = (
    "http://127.0.0.1:8000/score_batch"
)

def aesthetic_image_score(
    image_paths,
    precision=4,
    batch_size=8,
    timeout=300,
):

    try:

        response = requests.post(
            SERVER_URL,
            json={
                "image_paths": image_paths,
                "precision": precision,
                "batch_size": batch_size,
            },
            timeout=timeout,
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:

            raise RuntimeError(
                data["error"]
            )

        return [
            float(x)
            for x in data["scores"]
        ]

    except Exception as e:

        print(
            "Aesthetic score failed"
        )

        print(e)

        return [
            0.0
            for _ in image_paths
        ]