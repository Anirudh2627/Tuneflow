/** Typed API client for the TuneFlow backend. */
import type {
  Artist,
  ArtistDetail,
  EventType,
  Health,
  HistoryItem,
  LikedLibrary,
  RecMode,
  Recommendations,
  SearchResults,
  Session,
  SimilarSongs,
  Song,
  SongEvent,
  User,
  UserProfile,
} from '../types'

const BASE = '' // same-origin; Vite dev server proxies API paths to FastAPI

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* keep statusText */
    }
    throw new Error(`API ${response.status}: ${detail}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  // health
  health: () => request<Health>('/health'),

  // users
  createUser: (displayName: string, preferredGenres: string[] = []) =>
    request<User>('/users', {
      method: 'POST',
      body: JSON.stringify({ display_name: displayName, preferred_genres: preferredGenres }),
    }),
  getUser: (userId: string) => request<User>(`/users/${userId}`),
  getProfile: (userId: string) => request<UserProfile>(`/users/${userId}/profile`),
  getRecommendations: (userId: string, limit = 12, mode: RecMode = 'personalized', genre?: string) => {
    const params = new URLSearchParams({ limit: String(limit), mode })
    if (genre) params.set('genre', genre)
    return request<Recommendations>(`/users/${userId}/recommendations?${params}`)
  },
  getHistory: (userId: string, limit = 50, eventType?: EventType) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (eventType) params.set('event_type', eventType)
    return request<HistoryItem[]>(`/users/${userId}/history?${params}`)
  },
  getRecentlyPlayed: (userId: string, limit = 12) =>
    request<Song[]>(`/users/${userId}/recent?limit=${limit}`),
  getLiked: (userId: string) => request<LikedLibrary>(`/users/${userId}/liked`),

  // catalog
  getSong: (songId: string) => request<Song>(`/songs/${songId}`),
  getSimilarSongs: (songId: string, limit = 12) =>
    request<SimilarSongs>(`/songs/${songId}/similar?limit=${limit}`),
  getArtist: (artistId: string) => request<ArtistDetail>(`/artists/${artistId}`),
  listGenres: () => request<{ id: string; name: string }[]>('/genres'),
  getGenreSongs: (genreId: string, limit = 20) =>
    request<Song[]>(`/genres/${genreId}/songs?limit=${limit}`),
  search: (query: string, limit = 12) =>
    request<SearchResults>(`/search?q=${encodeURIComponent(query)}&limit=${limit}`),

  // events & feedback
  postEvent: (userId: string, songId: string, eventType: EventType, sessionId?: string | null) =>
    request<SongEvent>('/events', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        song_id: songId,
        event_type: eventType,
        session_id: sessionId ?? null,
      }),
    }),
  postFeedback: (userId: string, songId: string, feedback: EventType, sessionId?: string | null) =>
    request<SongEvent>('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        user_id: userId,
        song_id: songId,
        feedback,
        session_id: sessionId ?? null,
      }),
    }),

  // sessions
  createSession: (userId: string) =>
    request<Session>('/sessions', { method: 'POST', body: JSON.stringify({ user_id: userId }) }),
  getSession: (sessionId: string) => request<Session>(`/sessions/${sessionId}`),
  endSession: (sessionId: string) => request<Session>(`/sessions/${sessionId}/end`, { method: 'POST' }),
}
