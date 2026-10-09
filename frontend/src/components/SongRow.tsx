import { Link } from 'react-router-dom'
import type { RecommendedSong, Song } from '../types'
import { useApp } from '../store/AppContext'
import AlbumArt from './AlbumArt'
import FeedbackButtons from './FeedbackButtons'
import { formatDuration } from './SongCard'

export type RowItem = Song | RecommendedSong

function normalize(item: RowItem) {
  if ('song_id' in item) {
    return {
      id: item.song_id,
      title: item.title,
      artist: item.artist,
      artistId: item.artist_id,
      genres: item.genres,
      artwork: item.artwork_url,
      preview: item.preview_url,
      duration: item.duration_ms,
      reason: item.reason ?? null,
      raw: item,
    }
  }
  return {
    id: item.id,
    title: item.title,
    artist: item.artist_name,
    artistId: item.artist_id,
    genres: item.genres,
    artwork: item.artwork_url,
    preview: item.preview_url,
    duration: item.duration_ms,
    reason: null,
    raw: item,
  }
}

interface Props {
  item: RowItem
  index?: number
  onPlay?: () => void
}

/** Compact list row (history, library, song-page shelves). */
export default function SongRow({ item, index, onPlay }: Props) {
  const { playSong } = useApp()
  const song = normalize(item)

  return (
    <div
      className="group flex items-center gap-3 rounded-lg px-2 py-2 transition hover:bg-white/5"
      onDoubleClick={() => {
        playSong(song.raw as never)
        onPlay?.()
      }}
    >
      {index !== undefined && <span className="w-6 text-right text-xs text-zinc-500">{index + 1}</span>}
      <button
        onClick={() => {
          playSong(song.raw as never)
          onPlay?.()
        }}
        className="relative shrink-0"
        title="Play"
      >
        <AlbumArt id={song.id} title={song.title} imageUrl={song.artwork} size={44} />
        <span className="absolute inset-0 flex items-center justify-center rounded-lg bg-black/50 text-white opacity-0 transition group-hover:opacity-100">
          ▶
        </span>
      </button>
      <div className="min-w-0 flex-1">
        <Link to={`/songs/${song.id}`} className="block truncate text-sm font-medium hover:text-accent-soft">
          {song.title}
        </Link>
        <div className="flex items-center gap-2 truncate text-xs text-zinc-400">
          <Link to={`/artists/${song.artistId}`} className="truncate hover:text-white hover:underline">
            {song.artist}
          </Link>
          {song.genres[0] && <span className="chip shrink-0">{song.genres[0]}</span>}
        </div>
        {song.reason && (
          <p className="mt-0.5 truncate text-[11px] text-zinc-500" title={song.reason}>
            {song.reason}
          </p>
        )}
      </div>
      <span className="hidden text-xs text-zinc-500 sm:block">{formatDuration(song.duration)}</span>
      <FeedbackButtons songId={song.id} size="sm" />
    </div>
  )
}
