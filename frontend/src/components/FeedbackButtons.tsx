import { useState } from 'react'
import type { EventType } from '../types'
import { useApp } from '../store/AppContext'

interface Props {
  songId: string
  size?: 'sm' | 'md'
  onRecorded?: (eventType: EventType) => void
}

const BUTTONS: { event: EventType; icon: string; label: string; activeClass: string }[] = [
  { event: 'like', icon: '♥', label: 'Like', activeClass: 'text-emerald-400' },
  { event: 'dislike', icon: '✕', label: 'Dislike', activeClass: 'text-rose-400' },
  { event: 'save', icon: '⤓', label: 'Save', activeClass: 'text-sky-400' },
]

/** Like / Dislike / Save buttons wired to POST /feedback. */
export default function FeedbackButtons({ songId, size = 'md', onRecorded }: Props) {
  const { trackEvent, user } = useApp()
  const [active, setActive] = useState<Record<string, boolean>>({})
  const [pending, setPending] = useState<string | null>(null)

  const send = async (event: EventType) => {
    if (!user || pending) return
    setPending(event)
    await trackEvent(songId, event)
    setActive((prev) => ({ ...prev, [event]: !prev[event] }))
    setPending(null)
    onRecorded?.(event)
  }

  const dim = size === 'sm' ? 'text-xs px-1.5 py-1' : 'text-sm px-2 py-1.5'
  return (
    <div className="flex items-center gap-1">
      {BUTTONS.map((button) => (
        <button
          key={button.event}
          title={button.label}
          disabled={!user || pending !== null}
          onClick={(e) => {
            e.stopPropagation()
            void send(button.event)
          }}
          className={`rounded-full transition hover:bg-white/10 disabled:opacity-40 ${dim} ${
            active[button.event] ? button.activeClass : 'text-zinc-400 hover:text-white'
          }`}
        >
          {button.icon}
        </button>
      ))}
    </div>
  )
}
