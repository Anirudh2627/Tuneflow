import { useState } from 'react'
import { api } from '../services/api'
import type { RecMode } from '../types'
import { useApp } from '../store/AppContext'
import { useFetch } from '../hooks/useFetch'
import SongCard from '../components/SongCard'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

const MODES: { id: RecMode; label: string; hint: string }[] = [
  { id: 'personalized', label: 'Personalized', hint: 'Full hybrid ranking tuned to your taste' },
  { id: 'trending', label: 'Trending', hint: 'Popularity-weighted right now' },
  { id: 'discover', label: 'New Discoveries', hint: 'Novelty-boosted exploration' },
  { id: 'recent', label: 'Because You Listened', hint: 'Session-aware continuation' },
]

export default function DiscoverPage() {
  const { user } = useApp()
  const [mode, setMode] = useState<RecMode>('personalized')
  const [genre, setGenre] = useState<string | null>(null)

  const genres = useFetch(() => api.listGenres(), [])
  const recs = useFetch(
    user ? () => api.getRecommendations(user.id, 18, mode) : null, [user?.id, mode])
  const genreSongs = useFetch(
    genre ? () => api.getGenreSongs(genre, 18) : null, [genre])

  if (!user) return <EmptyState title="Create a profile first" />

  return (
    <div className="mx-auto max-w-6xl">
      <SectionHeader title="Discover" subtitle="Explore TuneFlow your way" />

      <div className="mb-6 flex flex-wrap gap-2">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setGenre(null) }}
            className={`chip cursor-pointer px-4 py-1.5 text-xs transition ${mode === m.id && !genre ? 'chip-active' : ''}`}
            title={m.hint}
          >
            {m.label}
          </button>
        ))}
      </div>

      <section className="mb-10">
        <SectionHeader
          title={genre ? `Top in ${genre}` : MODES.find((m) => m.id === mode)?.label ?? ''}
          subtitle={genre ? 'Ranked by listening activity' : MODES.find((m) => m.id === mode)?.hint}
        />
        {recs.error && <ErrorMessage message={recs.error} />}
        {(recs.loading || genreSongs.loading) && <Spinner />}
        {!genre && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
            {(recs.data?.items ?? []).map((song) => (
              <SongCard key={song.song_id} song={song} />
            ))}
          </div>
        )}
        {genre && (
          <div className="card divide-y divide-white/5 p-2">
            {(genreSongs.data ?? []).map((song, i) => (
              <SongRow key={song.id} item={song} index={i} />
            ))}
            {genreSongs.data?.length === 0 && <EmptyState title={`No tracks tagged ${genre} yet`} />}
          </div>
        )}
      </section>

      <section>
        <SectionHeader title="Browse by genre" subtitle="Catalog-wide genre shelves" />
        {genres.error && <ErrorMessage message={genres.error} />}
        <div className="flex flex-wrap gap-2">
          {(genres.data ?? []).map((g) => (
            <button
              key={g.id}
              onClick={() => setGenre(genre === g.id ? null : g.id)}
              className={`chip cursor-pointer px-4 py-1.5 text-xs transition ${genre === g.id ? 'chip-active' : ''}`}
            >
              {g.name}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}
