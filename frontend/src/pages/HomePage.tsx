import { Link } from 'react-router-dom'
import { api } from '../services/api'
import type { RecommendedSong } from '../types'
import { useApp } from '../store/AppContext'
import { useFetch } from '../hooks/useFetch'
import SectionHeader from '../components/SectionHeader'
import SongCard from '../components/SongCard'
import SongRow from '../components/SongRow'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

function Shelf({ title, subtitle, items, action }: {
  title: string
  subtitle?: string
  items: RecommendedSong[]
  action?: React.ReactNode
}) {
  if (!items || items.length === 0) return null
  return (
    <section className="mb-10">
      <SectionHeader title={title} subtitle={subtitle} action={action} />
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
        {items.map((song) => (
          <SongCard key={song.song_id} song={song} />
        ))}
      </div>
    </section>
  )
}

export default function HomePage() {
  const { user } = useApp()
  const userId = user?.id ?? null

  const madeForYou = useFetch(
    userId ? () => api.getRecommendations(userId, 12, 'personalized') : null, [userId])
  const recent = useFetch(userId ? () => api.getRecentlyPlayed(userId, 8) : null, [userId])
  const discover = useFetch(userId ? () => api.getRecommendations(userId, 6, 'discover') : null, [userId])
  const trending = useFetch(userId ? () => api.getRecommendations(userId, 6, 'trending') : null, [userId])
  const sessionBased = useFetch(userId ? () => api.getRecommendations(userId, 6, 'recent') : null, [userId])
  const liked = useFetch(userId ? () => api.getLiked(userId) : null, [userId])

  // "Because you liked <song>" — similar to the most recent like
  const seedLike = liked.data?.liked?.[0]
  const becauseYouLiked = useFetch(
    seedLike ? () => api.getSimilarSongs(seedLike.id, 6) : null, [seedLike?.id])

  if (!user) {
    return <EmptyState title="Welcome to TuneFlow" hint="Create a profile to start receiving recommendations." />
  }

  return (
    <div className="mx-auto max-w-6xl">
      <header className="mb-8">
        <p className="text-xs uppercase tracking-widest text-accent-soft">TuneFlow</p>
        <h1 className="mt-1 text-3xl font-extrabold tracking-tight">
          Hey {user.display_name.split(' ')[0]} 👋
        </h1>
        <p className="mt-1 text-sm text-zinc-400">
          {madeForYou.data?.strategy === 'hybrid'
            ? 'Hybrid engine active — personalized by content, collaborative, session & popularity signals.'
            : 'Running in starter mode — train the ML models to unlock full personalization.'}
        </p>
      </header>

      {madeForYou.error && <ErrorMessage message={madeForYou.error} />}
      {madeForYou.loading && !madeForYou.data && <Spinner label="Building your mix…" />}

      <Shelf
        title="Made For You"
        subtitle="Hybrid ranking over your taste profile"
        items={madeForYou.data?.items ?? []}
        action={<Link className="text-xs text-accent-soft hover:underline" to="/discover">See all</Link>}
      />

      {recent.data && recent.data.length > 0 && (
        <section className="mb-10">
          <SectionHeader title="Recently Played" subtitle="Pick up where you left off" />
          <div className="card divide-y divide-white/5 p-2">
            {recent.data.map((song, i) => (
              <SongRow key={song.id} item={song} index={i} />
            ))}
          </div>
        </section>
      )}

      <Shelf
        title={seedLike ? `Because You Liked “${seedLike.title}”` : 'Because You Liked…'}
        subtitle="Content similarity from tracks you loved"
        items={becauseYouLiked.data?.items ?? []}
      />

      <Shelf title="Discover Something New" subtitle="Novelty-boosted picks outside your routine"
             items={discover.data?.items ?? []} />

      <Shelf title="Trending" subtitle="What's hot on TuneFlow right now"
             items={trending.data?.items ?? []} />

      <Shelf title="Similar To Your Recent Listening" subtitle="Session-aware continuation of your vibe"
             items={sessionBased.data?.items ?? []} />
    </div>
  )
}
