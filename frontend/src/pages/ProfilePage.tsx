import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { useApp } from '../store/AppContext'
import { useFetch } from '../hooks/useFetch'
import SectionHeader from '../components/SectionHeader'
import { EmptyState, ErrorMessage, Spinner } from '../components/Status'

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card text-center">
      <p className="text-2xl font-extrabold text-accent-soft">{value}</p>
      <p className="mt-1 text-[11px] uppercase tracking-wide text-zinc-500">{label}</p>
    </div>
  )
}

function AffinityBars({ entries, emptyLabel }: {
  entries: [string, number][]
  emptyLabel: string
}) {
  const max = Math.max(...entries.map(([, v]) => v), 0.0001)
  if (entries.length === 0) return <p className="text-xs text-zinc-600">{emptyLabel}</p>
  return (
    <div className="space-y-2">
      {entries.map(([name, value]) => (
        <div key={name}>
          <div className="flex justify-between text-xs">
            <span className="text-zinc-300">{name}</span>
            <span className="text-zinc-500">{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-surface-700">
            <div className="h-full rounded-full bg-accent" style={{ width: `${(value / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function ProfilePage() {
  const { user, logout } = useApp()
  const profile = useFetch(user ? () => api.getProfile(user.id) : null, [user?.id])

  if (!user) return <EmptyState title="Create a profile first" />
  if (profile.loading && !profile.data) return <Spinner label="Building your taste profile…" />
  if (profile.error || !profile.data) return <ErrorMessage message={profile.error ?? 'Profile unavailable'} />

  const p = profile.data
  const stats = p.listening_stats
  const genreEntries = Object.entries(p.genre_affinity).sort((a, b) => b[1] - a[1]).slice(0, 6)
  const audioEntries = Object.entries(p.audio_feature_means)

  return (
    <div className="mx-auto max-w-4xl">
      <SectionHeader
        title={`Profile — ${p.display_name}`}
        subtitle={p.is_cold_start
          ? 'Cold start: we are still learning your taste. Like, save or skip a few tracks!'
          : 'Derived live from your interaction history'}
        action={<button className="btn btn-ghost text-xs" onClick={logout}>Switch user</button>}
      />

      <div className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Events" value={Math.round(stats.total_events ?? 0)} />
        <Stat label="Plays" value={Math.round(stats.positive_events ?? 0)} />
        <Stat label="Likes" value={Math.round(stats.likes ?? 0)} />
        <Stat label="Saves" value={Math.round(stats.saves ?? 0)} />
        <Stat label="Songs" value={Math.round(stats.unique_songs ?? 0)} />
        <Stat label="Artists" value={Math.round(stats.unique_artists ?? 0)} />
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <section className="card">
          <h3 className="mb-3 text-sm font-semibold text-zinc-300">Genre affinity</h3>
          <AffinityBars entries={genreEntries} emptyLabel="No genre signal yet." />
        </section>

        <section className="card">
          <h3 className="mb-3 text-sm font-semibold text-zinc-300">Top artists</h3>
          {p.top_artists.length === 0 && <p className="text-xs text-zinc-600">No artist signal yet.</p>}
          <div className="space-y-2">
            {p.top_artists.slice(0, 6).map((artist, i) => (
              <Link key={artist.artist_id} to={`/artists/${artist.artist_id}`}
                    className="flex items-center justify-between rounded-lg px-2 py-1.5 text-sm transition hover:bg-white/5">
                <span className="text-zinc-200">{i + 1}. {artist.name}</span>
                <span className="text-xs text-zinc-500">{(artist.affinity * 100).toFixed(1)}%</span>
              </Link>
            ))}
          </div>
        </section>

        <section className="card">
          <h3 className="mb-3 text-sm font-semibold text-zinc-300">Average audio profile</h3>
          {audioEntries.length === 0 ? (
            <p className="text-xs text-zinc-600">Not enough plays with audio features.</p>
          ) : (
            <div className="grid grid-cols-2 gap-x-6 gap-y-2">
              {audioEntries.map(([feature, value]) => (
                <div key={feature} className="flex justify-between text-xs">
                  <span className="capitalize text-zinc-400">{feature}</span>
                  <span className="font-mono text-zinc-200">
                    {feature === 'tempo' ? `${value.toFixed(0)} BPM`
                      : feature === 'loudness' ? `${value.toFixed(1)} dB`
                      : value.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="card">
          <h3 className="mb-3 text-sm font-semibold text-zinc-300">Moods & tags</h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(p.tag_affinity)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([tag, value]) => (
                <span key={tag} className="chip chip-active">
                  {tag} · {(value * 100).toFixed(0)}%
                </span>
              ))}
            {Object.keys(p.tag_affinity).length === 0 && (
              <p className="text-xs text-zinc-600">No tag signal yet.</p>
            )}
          </div>
          <h3 className="mb-2 mt-5 text-sm font-semibold text-zinc-300">Onboarding genres</h3>
          <div className="flex flex-wrap gap-2">
            {(user.preferred_genres ?? []).map((g) => <span key={g} className="chip">{g}</span>)}
            {(user.preferred_genres ?? []).length === 0 && (
              <p className="text-xs text-zinc-600">None selected.</p>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
