import { Link, useParams } from 'react-router-dom'
import { api } from '../services/api'
import { useFetch } from '../hooks/useFetch'
import { useApp } from '../store/AppContext'
import AlbumArt from '../components/AlbumArt'
import SongCard from '../components/SongCard'
import SongRow from '../components/SongRow'
import SectionHeader from '../components/SectionHeader'
import FeedbackButtons from '../components/FeedbackButtons'
import { ErrorMessage, Spinner } from '../components/Status'
import { formatDuration } from '../components/SongCard'

function FeatureBar({ label, value, display }: { label: string; value: number | null; display?: string }) {
  const pct = value === null ? 0 : Math.min(Math.max(value, 0), 1) * 100
  return (
    <div className="min-w-0">
      <div className="flex justify-between text-[10px] uppercase tracking-wide text-zinc-500">
        <span>{label}</span>
        <span>{display ?? (value === null ? '—' : value.toFixed(2))}</span>
      </div>
      <div className="mt-1 h-1 rounded-full bg-surface-700">
        <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export default function SongPage() {
  const { songId } = useParams<{ songId: string }>()
  const { playSong, user } = useApp()

  const song = useFetch(songId ? () => api.getSong(songId) : null, [songId])
  const similar = useFetch(songId ? () => api.getSimilarSongs(songId, 10) : null, [songId])
  const songData = song.data
  const artist = useFetch(songData ? () => api.getArtist(songData.artist_id) : null, [songData?.artist_id])
  const personalized = useFetch(
    user && songId ? () => api.getRecommendations(user.id, 6, 'recent') : null, [user?.id, songId])

  if (song.loading && !song.data) return <Spinner label="Loading track…" />
  if (song.error || !song.data) return <ErrorMessage message={song.error ?? 'Song not found'} />

  const s = song.data
  const moreFromArtist = (artist.data?.top_songs ?? []).filter((t) => t.id !== s.id).slice(0, 6)

  return (
    <div className="mx-auto max-w-5xl">
      <header className="mb-8 flex flex-col gap-5 sm:flex-row sm:items-end">
        <AlbumArt id={s.id} title={s.title} imageUrl={s.artwork_url} size={200} className="shadow-2xl" />
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-widest text-zinc-500">Song</p>
          <h1 className="mt-1 text-3xl font-extrabold tracking-tight md:text-4xl">{s.title}</h1>
          <p className="mt-2 text-sm text-zinc-300">
            <Link to={`/artists/${s.artist_id}`} className="font-semibold hover:text-accent-soft hover:underline">
              {s.artist_name}
            </Link>
            {s.album && <span className="text-zinc-500"> · {s.album}</span>}
            {s.release_year && <span className="text-zinc-500"> · {s.release_year}</span>}
            <span className="text-zinc-500"> · {formatDuration(s.duration_ms)}</span>
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {s.genres.map((g) => <span key={g} className="chip">{g}</span>)}
            {s.tags.map((t) => <span key={t} className="chip bg-surface-800">#{t}</span>)}
          </div>
          <div className="mt-4 flex items-center gap-3">
            <button className="btn btn-primary" onClick={() => playSong(s)}>▶ Play</button>
            <FeedbackButtons songId={s.id} />
          </div>
        </div>
      </header>

      <section className="card mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
        <FeatureBar label="Energy" value={s.energy} />
        <FeatureBar label="Danceability" value={s.danceability} />
        <FeatureBar label="Valence" value={s.valence} />
        <FeatureBar label="Acousticness" value={s.acousticness} />
        <FeatureBar label="Tempo" value={s.tempo ? Math.min(s.tempo / 200, 1) : null}
                    display={s.tempo ? `${Math.round(s.tempo)} BPM` : undefined} />
        <FeatureBar label="Instrumental" value={s.instrumentalness} />
        <FeatureBar label="Speechiness" value={s.speechiness} />
        <FeatureBar label="Popularity" value={s.popularity !== null ? s.popularity / 100 : null}
                    display={s.popularity !== null ? String(Math.round(s.popularity)) : undefined} />
      </section>

      <section className="mb-10">
        <SectionHeader title="Similar Songs" subtitle="Content-based similarity in audio & metadata space" />
        {similar.loading && <Spinner />}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {(similar.data?.items ?? []).map((item) => <SongCard key={item.song_id} song={item} />)}
        </div>
      </section>

      {personalized.data && personalized.data.items.length > 0 && (
        <section className="mb-10">
          <SectionHeader title="Because You Listened To…" subtitle="Session-aware picks from your recent vibe" />
          <div className="card divide-y divide-white/5 p-2">
            {personalized.data.items.map((item, i) => <SongRow key={item.song_id} item={item} index={i} />)}
          </div>
        </section>
      )}

      {moreFromArtist.length > 0 && (
        <section className="mb-10">
          <SectionHeader
            title={`More From ${s.artist_name}`}
            action={<Link to={`/artists/${s.artist_id}`} className="text-xs text-accent-soft hover:underline">
              View artist</Link>}
          />
          <div className="card divide-y divide-white/5 p-2">
            {moreFromArtist.map((item, i) => <SongRow key={item.id} item={item} index={i} />)}
          </div>
        </section>
      )}
    </div>
  )
}
