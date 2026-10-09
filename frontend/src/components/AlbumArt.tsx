/**
 * Deterministic generated album art.
 *
 * The sample/MSD datasets ship no imagery, so we render a stable
 * gradient + music glyph derived from a hash of the song id.  Real
 * artwork_url values always take precedence.
 */
import { useMemo } from 'react'

function hashString(value: string): number {
  let hash = 2166136261
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

interface Props {
  id: string
  title?: string
  imageUrl?: string | null
  size?: number
  className?: string
  rounded?: string
}

export default function AlbumArt({ id, title, imageUrl, size = 56, className = '', rounded = 'rounded-lg' }: Props) {
  const { hueA, hueB, angle } = useMemo(() => {
    const h = hashString(id)
    return {
      hueA: h % 360,
      hueB: (h >> 9) % 360,
      angle: (h >> 17) % 360,
    }
  }, [id])

  if (imageUrl) {
    return (
      <img
        src={imageUrl}
        alt={title ?? id}
        width={size}
        height={size}
        className={`${rounded} object-cover ${className}`}
        style={{ width: size, height: size }}
      />
    )
  }

  const initial = (title ?? id).charAt(0).toUpperCase()
  return (
    <div
      className={`relative flex items-center justify-center overflow-hidden ${rounded} ${className}`}
      style={{
        width: size,
        height: size,
        minWidth: size,
        background: `linear-gradient(${angle}deg, hsl(${hueA} 70% 35%), hsl(${hueB} 75% 22%))`,
      }}
      role="img"
      aria-label={title ?? id}
    >
      <span className="text-white/85 font-bold" style={{ fontSize: Math.max(size * 0.38, 12) }}>
        {initial}
      </span>
      <svg
        className="absolute bottom-1 right-1 text-white/25"
        width={Math.max(size * 0.3, 12)}
        height={Math.max(size * 0.3, 12)}
        viewBox="0 0 24 24"
        fill="currentColor"
        aria-hidden
      >
        <path d="M9 18V5l12-2v13" stroke="currentColor" strokeWidth="2" fill="none" />
        <circle cx="6" cy="18" r="3" />
        <circle cx="18" cy="16" r="3" />
      </svg>
    </div>
  )
}
