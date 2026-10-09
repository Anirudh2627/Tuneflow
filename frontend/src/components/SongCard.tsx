import { Link } from 'react-router-dom'
import type { RecommendedSong } from '../types'
import { useApp } from '../store/AppContext'
import AlbumArt from './AlbumArt'
import FeedbackButtons from './FeedbackButtons'

export function formatDuration(ms: number | null | undefined): string {
  if (!ms) return '—'
  const total = Math.round(ms / 1000)
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${s.toString().padStart(2, '0')}`
}

interface Props {
  song: RecommendedSong
  showReason?: boolean
}

/** Grid card used by Home / Discover shelves. */
export default function SongCard({ song, showReason = true }: Props) {
  const { playSong } = useApp()

  return (
    <div className="card group relative flex flex-col gap-3 transition hover:border-accent/30">
      <div className="relative">
        <Link to={`/songs/${song.song_id}`}>
          <AlbumArt
            id={song.song_id}
            title={song.title}
            imageUrl={song.artwork_url}
            size={160}
            className="w-full"
            rounded="rounded-lg"
          />
        </Link>
        <button
          onClick={() => playSong(song)}
          className="absolute bottom-2 right-2 flex h-10 w-10 items-center justify-center rounded-full
                     bg-accent text-white opacity-0 shadow-lg transition group-hover:opacity-100 hover:scale-105"
          title="Play"
        >
          ▶
        </button>
      </div>

      <div className="min-w-0">
        <Link to={`/songs/${song.song_id}`} className="block truncate text-sm font-semibold hover:text-accent-soft">
          {song.title}
        </Link>
        <Link to={`/artists/${song.artist_id}`} className="block truncate text-xs text-zinc-400 hover:text-white">
          {song.artist}
        </Link>
        <div className="mt-1.5 flex flex-wrap items-center gap-1">
          {song.genres.slice(0, 2).map((genre) => (
            <span key={genre} className="chip">{genre}</span>
          ))}
        </div>
      </div>

      {showReason && song.reason && (
        <p className="line-clamp-2 text-[11px] leading-snug text-zinc-500" title={song.reason}>
          {song.reason}
        </p>
      )}

      <div className="mt-auto flex items-center justify-between">
        <span className="text-[11px] text-zinc-600">score {song.score.toFixed(2)}</span>
        <FeedbackButtons songId={song.song_id} size="sm" />
      </div>
    </div>
  )
}
