import { useState } from 'react'
import { api } from '../services/api'
import type { EventType } from '../types'
import { useApp } from '../store/AppContext'
import { useFetch } from '../hooks/useFetch'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

const FILTERS: { id: EventType | 'all'; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'play', label: 'Plays' },
  { id: 'like', label: 'Likes' },
  { id: 'save', label: 'Saves' },
  { id: 'skip', label: 'Skips' },
  { id: 'dislike', label: 'Dislikes' },
]

function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const minutes = Math.floor(diff / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(iso).toLocaleDateString()
}

export default function HistoryPage() {
  const { user } = useApp()
  const [filter, setFilter] = useState<EventType | 'all'>('all')

  const history = useFetch(
    user ? () => api.getHistory(user.id, 100, filter === 'all' ? undefined : filter) : null,
    [user?.id, filter])

  if (!user) return <EmptyState title="Create a profile first" />

  return (
    <div className="mx-auto max-w-4xl">
      <SectionHeader title="Listening History" subtitle="Every event feeds the next training run" />

      <div className="mb-5 flex flex-wrap gap-2">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            className={`chip cursor-pointer px-4 py-1.5 text-xs transition ${filter === f.id ? 'chip-active' : ''}`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {history.loading && <Spinner />}
      {history.error && <ErrorMessage message={history.error} />}
      {history.data && history.data.length === 0 && (
        <EmptyState title="No events yet" hint="Play a song and it will show up here." />
      )}
      {history.data && history.data.length > 0 && (
        <div className="card divide-y divide-white/5 p-2">
          {history.data.map((item) => (
            <div key={item.id} className="flex items-center gap-2">
              <div className="min-w-0 flex-1">
                <SongRow item={item.song} />
              </div>
              <div className="hidden w-28 shrink-0 text-right sm:block">
                <span className={`chip ${
                  item.event_type === 'like' ? 'text-emerald-400'
                    : item.event_type === 'dislike' ? 'text-rose-400'
                    : item.event_type === 'skip' ? 'text-zinc-500'
                    : item.event_type === 'save' ? 'text-sky-400' : ''
                }`}>
                  {item.event_type}
                </span>
                <p className="mt-1 text-[10px] text-zinc-600">{timeAgo(item.timestamp)}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
