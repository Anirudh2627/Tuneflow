/**
 * Global app state: current user (persisted in localStorage), the audio
 * player queue and the active session id.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api } from '../services/api'
import type { RecommendedSong, Song, User } from '../types'

export interface PlayerItem {
  song_id: string
  title: string
  artist: string
  preview_url: string | null
  artwork_url: string | null
  duration_ms: number | null
}

interface AppState {
  user: User | null
  ready: boolean
  onboardingOpen: boolean
  openOnboarding: () => void
  closeOnboarding: () => void
  loginAsDemo: () => Promise<void>
  finishOnboarding: (name: string, genres: string[]) => Promise<void>
  logout: () => void
  refreshUser: () => Promise<void>

  playerItem: PlayerItem | null
  playSong: (item: PlayerItem | Song | RecommendedSong) => void
  stopPlayer: () => void

  sessionId: string | null
  trackEvent: (songId: string, eventType: 'play' | 'like' | 'dislike' | 'save' | 'skip') => Promise<void>
}

const AppContext = createContext<AppState | null>(null)

const USER_STORAGE_KEY = 'tuneflow.user'

function toPlayerItem(item: PlayerItem | Song | RecommendedSong): PlayerItem {
  if ('song_id' in item) {
    return {
      song_id: item.song_id,
      title: item.title,
      artist: item.artist,
      preview_url: item.preview_url,
      artwork_url: item.artwork_url,
      duration_ms: item.duration_ms,
    }
  }
  return {
    song_id: item.id,
    title: item.title,
    artist: item.artist_name,
    preview_url: item.preview_url,
    artwork_url: item.artwork_url,
    duration_ms: item.duration_ms,
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [ready, setReady] = useState(false)
  const [onboardingOpen, setOnboardingOpen] = useState(false)
  const [playerItem, setPlayerItem] = useState<PlayerItem | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)

  // restore persisted user
  useEffect(() => {
    const stored = localStorage.getItem(USER_STORAGE_KEY)
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as User
        api
          .getUser(parsed.id)
          .then((fresh) => {
            setUser(fresh)
            localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(fresh))
          })
          .catch(() => localStorage.removeItem(USER_STORAGE_KEY))
          .finally(() => setReady(true))
        return
      } catch {
        localStorage.removeItem(USER_STORAGE_KEY)
      }
    }
    setReady(true)
    setOnboardingOpen(true)
  }, [])

  // ensure an active session whenever a user exists
  useEffect(() => {
    if (!user) {
      setSessionId(null)
      return
    }
    api
      .createSession(user.id)
      .then((session) => setSessionId(session.id))
      .catch(() => setSessionId(null))
  }, [user?.id])

  const persistUser = useCallback((next: User | null) => {
    setUser(next)
    if (next) localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(next))
    else localStorage.removeItem(USER_STORAGE_KEY)
  }, [])

  const loginAsDemo = useCallback(async () => {
    try {
      const demo = await api.getUser('usr_demo')
      persistUser(demo)
      setOnboardingOpen(false)
    } catch {
      // demo user missing -> seed the DB (python scripts/seed_db.py)
      const created = await api.createUser('Demo Listener', ['electronic', 'indie'])
      persistUser(created)
      setOnboardingOpen(false)
    }
  }, [persistUser])

  const finishOnboarding = useCallback(
    async (name: string, genres: string[]) => {
      const created = await api.createUser(name, genres)
      persistUser(created)
      setOnboardingOpen(false)
    },
    [persistUser],
  )

  const logout = useCallback(() => {
    persistUser(null)
    setPlayerItem(null)
    setOnboardingOpen(true)
  }, [persistUser])

  const refreshUser = useCallback(async () => {
    if (!user) return
    const fresh = await api.getUser(user.id)
    persistUser(fresh)
  }, [user, persistUser])

  const trackEvent = useCallback(
    async (songId: string, eventType: 'play' | 'like' | 'dislike' | 'save' | 'skip') => {
      if (!user) return
      try {
        await api.postFeedback(user.id, songId, eventType, sessionId)
      } catch (error) {
        console.error('failed to track event', error)
      }
    },
    [user, sessionId],
  )

  const playSong = useCallback(
    (item: PlayerItem | Song | RecommendedSong) => {
      const player = toPlayerItem(item)
      setPlayerItem(player)
      void trackEvent(player.song_id, 'play')
    },
    [trackEvent],
  )

  const value = useMemo<AppState>(
    () => ({
      user,
      ready,
      onboardingOpen,
      openOnboarding: () => setOnboardingOpen(true),
      closeOnboarding: () => setOnboardingOpen(false),
      loginAsDemo,
      finishOnboarding,
      logout,
      refreshUser,
      playerItem,
      playSong,
      stopPlayer: () => setPlayerItem(null),
      sessionId,
      trackEvent,
    }),
    [user, ready, onboardingOpen, loginAsDemo, finishOnboarding, logout, refreshUser,
     playerItem, playSong, sessionId, trackEvent],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppState {
  const context = useContext(AppContext)
  if (!context) throw new Error('useApp must be used inside <AppProvider>')
  return context
}
