from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def register(email: str) -> dict:
    response = client.post(
        "/auth/register",
        json={"email": email, "username": email.split("@")[0], "password": "correct-horse-battery-staple"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def auth_headers(token_response: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {token_response['access_token']}"}


def test_project_routes_require_authentication():
    response = client.get("/projects")
    assert response.status_code == 401


def test_project_data_is_scoped_to_authenticated_owner():
    alice = register("alice@example.com")
    bob = register("bob@example.com")

    create_response = client.post(
        "/projects",
        headers=auth_headers(alice),
        json={"name": "Alice project", "task_type": "OBJECT_DETECTION", "owner_id": 999999},
    )
    assert create_response.status_code == 200, create_response.text
    project = create_response.json()
    assert project["owner_id"] == alice["user"]["user_id"]

    assert client.get("/projects", headers=auth_headers(alice)).json() == [project]
    assert client.get("/projects", headers=auth_headers(bob)).json() == []
    assert client.get(f"/projects/{project['project_id']}", headers=auth_headers(bob)).status_code == 403
    assert client.get(
        f"/images/project/{project['project_id']}", headers=auth_headers(bob)
    ).status_code == 403


def test_refresh_tokens_rotate_and_logout_revokes_session():
    account = register("refresh@example.com")
    refresh_response = client.post("/auth/refresh", json={"refresh_token": account["refresh_token"]})
    assert refresh_response.status_code == 200, refresh_response.text
    rotated = refresh_response.json()

    reused_response = client.post("/auth/refresh", json={"refresh_token": account["refresh_token"]})
    assert reused_response.status_code == 401

    logout_response = client.post("/auth/logout", json={"refresh_token": rotated["refresh_token"]})
    assert logout_response.status_code == 204
    assert client.post("/auth/refresh", json={"refresh_token": rotated["refresh_token"]}).status_code == 401
