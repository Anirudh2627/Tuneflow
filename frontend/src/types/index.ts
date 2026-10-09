/** Shared API types mirroring the backend Pydantic schemas. */

export type EventType = 'play' | 'skip' | 'like' | 'dislike' | 'save'
export type RecMode = 'personalized' | 'discover' | 'recent' | 'trending'

export interface Song {
  id: string
  title: string
  artist_id: string
  artist_name: string
  album: string | null
  album_id: string | null
  release_year: number | null
  duration_ms: number | null
  genres: string[]
  tags: string[]
  preview_url: string | null
  artwork_url: string | null
  energy: number | null
  danceability: number | null
  valence: number | null
  acousticness: number | null
  instrumentalness: number | null
  speechiness: number | null
  loudness: number | null
  tempo: number | null
  popularity: number | null
}

export interface Artist {
  id: string
  name: string
  genres: string[]
  followers: number
  image_url: string | null
}

export interface Album {
  id: string
  title: string
  artist_id: string
  release_year: number | null
  image_url: string | null
}

export interface ArtistDetail extends Artist {
  top_songs: Song[]
  albums: Album[]
}

export interface RecommendedSong {
  song_id: string
  title: string
  artist: string
  artist_id: string
  album: string | null
  genres: string[]
  artwork_url: string | null
  preview_url: string | null
  duration_ms: number | null
  score: number
  reason: string
  signals: string[]
}

export interface Recommendations {
  user_id: string
  mode: RecMode
  generated_at: string
  strategy: 'hybrid' | 'popularity-fallback'
  items: RecommendedSong[]
}

export interface SimilarSongs {
  song_id: string
  items: RecommendedSong[]
}

export interface User {
  id: string
  display_name: string
  email: string | null
  preferred_genres: string[]
  created_at: string
  last_active_at: string | null
}

export interface UserProfile {
  user_id: string
  display_name: string
  top_genres: string[]
  genre_affinity: Record<string, number>
  top_artists: { artist_id: string; name: string; affinity: number }[]
  tag_affinity: Record<string, number>
  audio_feature_means: Record<string, number>
  liked_songs: string[]
  disliked_songs: string[]
  saved_songs: string[]
  recent_songs: string[]
  listening_stats: Record<string, number>
  is_cold_start: boolean
}

export interface SongEvent {
  id: string
  user_id: string
  song_id: string
  event_type: EventType
  timestamp: string
  session_id: string | null
  weight: number
}

export interface HistoryItem {
  id: string
  event_type: EventType
  timestamp: string
  session_id: string | null
  weight: number
  song: Song
}

export interface Session {
  id: string
  user_id: string
  start_time: string
  last_activity: string
  ended: boolean
  song_ids: string[]
  events: SongEvent[]
}

export interface SearchResults {
  query: string
  songs: Song[]
  artists: Artist[]
  albums: Album[]
  genres: { id: string; name: string }[]
}

export interface LikedLibrary {
  user_id: string
  liked: Song[]
  saved: Song[]
  disliked: Song[]
}

export interface Health {
  status: string
  engine: 'hybrid' | 'degraded-popularity-fallback'
  components: Record<string, boolean>
  degraded_reason: string | null
}
