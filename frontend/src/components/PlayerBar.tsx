/**
 * Bottom audio player.
 *
 * Plays `preview_url` when the catalog provides one (e.g. FMA-style
 * datasets with legal preview clips).  Recommendation functionality is
 * fully independent of playback: without a preview URL the player still
 * shows the current track and records PLAY events.
 */
import { useEffect, useRef, useState } from 'react'
import { useApp } from '../store/AppContext'
import AlbumArt from './AlbumArt'
import { formatDuration } from './SongCard'

function fmt(seconds: number): string {
  if (!isFinite(seconds)) return '0:00'
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

export default function PlayerBar() {
  const { playerItem, stopPlayer } = useApp()
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const [duration, setDuration] = useState(0)

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || !playerItem) return
    if (playerItem.preview_url) {
      audio.src = playerItem.preview_url
      audio.play().then(() => setPlaying(true)).catch(() => setPlaying(false))
    } else {
      audio.removeAttribute('src')
      setPlaying(false)
      setPosition(0)
      setDuration((playerItem.duration_ms ?? 0) / 1000)
    }
  }, [playerItem])

  if (!playerItem) return null

  const toggle = () => {
    const audio = audioRef.current
    if (!audio || !playerItem.preview_url) return
    if (audio.paused) {
      void audio.play()
      setPlaying(true)
    } else {
      audio.pause()
      setPlaying(false)
    }
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 border-t border-white/10 bg-surface-900/95 backdrop-blur md:left-56">
      <audio
        ref={audioRef}
        onTimeUpdate={(e) => setPosition(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onEnded={() => setPlaying(false)}
      />
      <div className="mx-auto flex max-w-6xl items-center gap-4 px-4 py-3">
        <AlbumArt id={playerItem.song_id} title={playerItem.title}
                  imageUrl={playerItem.artwork_url} size={48} />
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold">{playerItem.title}</p>
          <p className="truncate text-xs text-zinc-400">{playerItem.artist}</p>
          <div className="mt-1 flex items-center gap-2">
            <span className="text-[10px] text-zinc-500">{fmt(position)}</span>
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-surface-700">
              <div
                className="h-full bg-accent"
                style={{ width: `${duration ? (position / duration) * 100 : 0}%` }}
              />
            </div>
            <span className="text-[10px] text-zinc-500">
              {duration ? fmt(duration) : formatDuration(playerItem.duration_ms)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggle}
            disabled={!playerItem.preview_url}
            className="flex h-10 w-10 items-center justify-center rounded-full bg-accent text-white
                       transition hover:bg-accent-soft disabled:opacity-40"
            title={playerItem.preview_url ? (playing ? 'Pause' : 'Play') : 'No audio preview in this dataset'}
          >
            {playing ? '❚❚' : '▶'}
          </button>
          <button onClick={stopPlayer} className="btn-ghost btn px-2 text-xs" title="Close player">
            ✕
          </button>
        </div>
      </div>
      {!playerItem.preview_url && (
        <p className="pb-2 text-center text-[10px] text-zinc-600">
          No playable preview for this track — recommendations work independently of playback.
        </p>
      )}
    </div>
  )
}
