"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-01-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("display_name", sa.String(120), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("preferred_genres", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "artists",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("genres", sa.JSON(), nullable=True),
        sa.Column("followers", sa.Float(), nullable=False, server_default="0"),
        sa.Column("image_url", sa.String(512), nullable=True),
    )
    op.create_index("ix_artists_name", "artists", ["name"])

    op.create_table(
        "albums",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("artist_id", sa.String(64),
                  sa.ForeignKey("artists.id", ondelete="CASCADE"), nullable=False),
        sa.Column("release_year", sa.Integer(), nullable=True),
        sa.Column("image_url", sa.String(512), nullable=True),
    )
    op.create_index("ix_albums_artist_id", "albums", ["artist_id"])

    op.create_table(
        "genres",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
    )

    op.create_table(
        "songs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("artist_id", sa.String(64),
                  sa.ForeignKey("artists.id", ondelete="CASCADE"), nullable=False),
        sa.Column("album_id", sa.String(64),
                  sa.ForeignKey("albums.id", ondelete="SET NULL"), nullable=True),
        sa.Column("release_year", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("preview_url", sa.String(512), nullable=True),
        sa.Column("artwork_url", sa.String(512), nullable=True),
        sa.Column("energy", sa.Float(), nullable=True),
        sa.Column("danceability", sa.Float(), nullable=True),
        sa.Column("valence", sa.Float(), nullable=True),
        sa.Column("acousticness", sa.Float(), nullable=True),
        sa.Column("instrumentalness", sa.Float(), nullable=True),
        sa.Column("speechiness", sa.Float(), nullable=True),
        sa.Column("loudness", sa.Float(), nullable=True),
        sa.Column("tempo", sa.Float(), nullable=True),
        sa.Column("popularity", sa.Float(), nullable=True),
    )
    op.create_index("ix_songs_title", "songs", ["title"])
    op.create_index("ix_songs_artist_id", "songs", ["artist_id"])

    op.create_table(
        "song_genres",
        sa.Column("song_id", sa.String(64),
                  sa.ForeignKey("songs.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("genre_id", sa.String(64),
                  sa.ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "user_song_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("song_id", sa.String(64),
                  sa.ForeignKey("songs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("event_type", sa.String(16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_id", sa.String(64),
                  sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1"),
    )
    op.create_index("ix_user_song_events_user_id", "user_song_events", ["user_id"])
    op.create_index("ix_user_song_events_song_id", "user_song_events", ["song_id"])
    op.create_index("ix_user_song_events_timestamp", "user_song_events", ["timestamp"])
    op.create_index("ix_user_song_events_session_id", "user_song_events", ["session_id"])

    op.create_table(
        "recommendations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False, server_default="personalized"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("items", sa.JSON(), nullable=True),
        sa.Column("context", sa.JSON(), nullable=True),
    )
    op.create_index("ix_recommendations_user_id", "recommendations", ["user_id"])
    op.create_index("ix_recommendations_created_at", "recommendations", ["created_at"])


def downgrade() -> None:
    for table in ("recommendations", "user_song_events", "sessions", "song_genres",
                  "songs", "genres", "albums", "artists", "users"):
        op.drop_table(table)
