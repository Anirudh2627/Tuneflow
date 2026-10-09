import { useState } from 'react'
import { api } from '../services/api'
import type { Song } from '../types'
import { useApp } from '../store/AppContext'
import { useFetch } from '../hooks/useFetch'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

type Tab = 'liked' | 'saved' | 'disliked'

const TABS: { id: Tab; label: string }[] = [
  { id: 'liked', label: '♥ Liked' },
  { id: 'saved', label: '⤓ Saved' },
  { id: 'disliked', label: '✕ Disliked' },
]

export default function LibraryPage() {
  const { user } = useApp()
  const [tab, setTab] = useState<Tab>('liked')
  const library = useFetch(user ? () => api.getLiked(user.id) : null, [user?.id])

  if (!user) return <EmptyState title="Create a profile first" />
  if (library.loading && !library.data) return <Spinner label="Loading your library…" />
  if (library.error) return <ErrorMessage message={library.error} />

  const items: Song[] = library.data?.[tab] ?? []

  return (
    <div className="mx-auto max-w-4xl">
      <SectionHeader
        title="Your Library"
        subtitle="Feedback you gave TuneFlow — it directly shapes future recommendations"
      />
      <div className="mb-5 flex gap-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`chip cursor-pointer px-4 py-1.5 text-xs transition ${tab === t.id ? 'chip-active' : ''}`}
          >
            {t.label} ({library.data?.[t.id]?.length ?? 0})
          </button>
        ))}
      </div>

      {items.length === 0 ? (
        <EmptyState
          title={`Nothing ${tab} yet`}
          hint="Use the ♥ / ✕ / ⤓ buttons on any song — your feedback loops back into the recommender."
        />
      ) : (
        <div className="card divide-y divide-white/5 p-2">
          {items.map((song, i) => <SongRow key={song.id} item={song} index={i} />)}
        </div>
      )}
    </div>
  )
}
