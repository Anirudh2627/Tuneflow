"""API endpoint tests (users, catalog, recommendations, feedback, sessions)."""
from __future__ import annotations


# ---------------------------------------------------------------- health
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"] in ("hybrid", "degraded-popularity-fallback")


# ---------------------------------------------------------------- users
def test_create_user(client):
    response = client.post("/users", json={"display_name": "Bob", "preferred_genres": ["jazz"]})
    assert response.status_code == 201
    body = response.json()
    assert body["display_name"] == "Bob"
    assert body["preferred_genres"] == ["jazz"]
    assert body["id"].startswith("usr_")


def test_create_user_validation(client):
    assert client.post("/users", json={}).status_code == 422
    assert client.post("/users", json={"display_name": ""}).status_code == 422


def test_get_user_and_404(client):
    created = client.post("/users", json={"display_name": "Carol"}).json()
    assert client.get(f"/users/{created['id']}").status_code == 200
    response = client.get("/users/usr_missing")
    assert response.status_code == 404


# ---------------------------------------------------------------- recommendations
def test_recommendations_cold_start(client):
    created = client.post("/users", json={"display_name": "Newbie",
                                          "preferred_genres": ["electronic"]}).json()
    response = client.get(f"/users/{created['id']}/recommendations?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "personalized"
    assert len(body["items"]) > 0
    first = body["items"][0]
    # every recommendation must carry an explanation + signals
    assert first["reason"]
    assert isinstance(first["signals"], list) and first["signals"]
    assert first["score"] > 0


def test_recommendations_modes_and_limits(client, demo_events):
    for mode in ("personalized", "discover", "recent", "trending"):
        response = client.get(f"/users/usr_demo/recommendations?mode={mode}&limit=3")
        assert response.status_code == 200, mode
        assert len(response.json()["items"]) <= 3
    assert client.get("/users/usr_demo/recommendations?mode=bogus").status_code == 422
    assert client.get("/users/usr_demo/recommendations?limit=1000").status_code == 422


def test_recommendations_exclude_history(client, demo_events):
    response = client.get("/users/usr_demo/recommendations?limit=20")
    recommended = {item["song_id"] for item in response.json()["items"]}
    # songs the demo user played/liked must not be recommended back
    assert not (recommended & {"s00", "s01", "s02", "s03"})


def test_recommendations_unknown_user(client):
    assert client.get("/users/usr_ghost/recommendations").status_code == 404


# ---------------------------------------------------------------- catalog
def test_get_song(client):
    response = client.get("/songs/s00")
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Track 0"
    assert body["artist_name"] == "Pulse Machine"
    assert body["genres"] == ["electronic"]
    assert client.get("/songs/missing").status_code == 404


def test_similar_songs(client):
    response = client.get("/songs/s00/similar?limit=5")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) > 0
    assert all(item["song_id"] != "s00" for item in items)
    assert client.get("/songs/nope/similar").status_code == 404


def test_artist_detail(client):
    response = client.get("/artists/ar1")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Pulse Machine"
    assert len(body["top_songs"]) > 0
    assert client.get("/artists/missing").status_code == 404


def test_search(client):
    response = client.get("/search?q=Track 1")
    assert response.status_code == 200
    body = response.json()
    assert len(body["songs"]) >= 1
    response = client.get("/search?q=Pulse")
    assert len(response.json()["artists"]) == 1
    response = client.get("/search?q=jazz")
    assert len(response.json()["genres"]) == 1
    assert client.get("/search?q=").status_code == 422


# ---------------------------------------------------------------- events & feedback
def test_post_event_creates_session_and_history(client):
    response = client.post("/events", json={
        "user_id": "usr_alice", "song_id": "s05", "event_type": "play"})
    assert response.status_code == 201
    event = response.json()
    assert event["session_id"] is not None
    assert event["weight"] == 1.0

    history = client.get("/users/usr_alice/history").json()
    assert len(history) == 1
    assert history[0]["song"]["id"] == "s05"
    assert history[0]["event_type"] == "play"


def test_event_validation(client):
    bad_type = client.post("/events", json={
        "user_id": "usr_alice", "song_id": "s05", "event_type": "explode"})
    assert bad_type.status_code == 422
    assert client.post("/events", json={
        "user_id": "usr_ghost", "song_id": "s05", "event_type": "play"}).status_code == 404
    assert client.post("/events", json={
        "user_id": "usr_alice", "song_id": "nope", "event_type": "play"}).status_code == 404


def test_feedback_weights(client):
    like = client.post("/feedback", json={
        "user_id": "usr_alice", "song_id": "s01", "feedback": "like"}).json()
    dislike = client.post("/feedback", json={
        "user_id": "usr_alice", "song_id": "s02", "feedback": "dislike"}).json()
    assert like["weight"] > 0 > dislike["weight"]

    recent = client.get("/users/usr_alice/recent").json()
    assert isinstance(recent, list)

    liked = client.get("/users/usr_alice/liked").json()
    assert [s["id"] for s in liked["liked"]] == ["s01"]
    assert [s["id"] for s in liked["disliked"]] == ["s02"]


def test_batch_events(client):
    payload = {"events": [
        {"user_id": "usr_alice", "song_id": "s03", "event_type": "play"},
        {"user_id": "usr_alice", "song_id": "s04", "event_type": "like"},
    ]}
    response = client.post("/events/batch", json=payload)
    assert response.status_code == 201
    assert len(response.json()) == 2


# ---------------------------------------------------------------- sessions
def test_session_lifecycle(client):
    created = client.post("/sessions", json={"user_id": "usr_alice"})
    assert created.status_code == 201
    session = created.json()
    assert session["user_id"] == "usr_alice"
    assert session["ended"] is False

    client.post("/events", json={
        "user_id": "usr_alice", "song_id": "s06", "event_type": "play",
        "session_id": session["id"]})

    fetched = client.get(f"/sessions/{session['id']}").json()
    assert fetched["song_ids"] == ["s06"]
    assert len(fetched["events"]) == 1

    ended = client.post(f"/sessions/{session['id']}/end").json()
    assert ended["ended"] is True

    assert client.get("/sessions/missing").status_code == 404
    assert client.post("/sessions", json={"user_id": "usr_ghost"}).status_code == 404


# ---------------------------------------------------------------- profile
def test_user_profile(client, demo_events):
    response = client.get("/users/usr_demo/profile")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "usr_demo"
    assert body["is_cold_start"] is False
    assert "electronic" in body["top_genres"]
    assert body["listening_stats"]["likes"] >= 1
    assert "s01" in body["liked_songs"]
    assert "s10" in body["disliked_songs"]


def test_profile_cold_start(client):
    created = client.post("/users", json={"display_name": "Empty"}).json()
    body = client.get(f"/users/{created['id']}/profile").json()
    assert body["is_cold_start"] is True
    assert body["top_genres"] == []
