import { NavLink, Outlet } from 'react-router-dom'
import PlayerBar from './PlayerBar'
import { useApp } from '../store/AppContext'

const NAV = [
  { to: '/', label: 'Home', icon: '🏠' },
  { to: '/discover', label: 'Discover', icon: '🧭' },
  { to: '/search', label: 'Search', icon: '🔍' },
  { to: '/library', label: 'Library', icon: '📚' },
  { to: '/history', label: 'History', icon: '🕘' },
  { to: '/profile', label: 'Profile', icon: '👤' },
]

export default function Layout() {
  const { user, logout, openOnboarding } = useApp()

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="hidden w-56 shrink-0 flex-col border-r border-white/5 bg-surface-900 p-4 md:flex">
        <div className="mb-8 flex items-center gap-2 px-2">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="8" fill="#8b5cf6" />
            <path d="M11 22V10l11-2v12" stroke="white" strokeWidth="2.4" fill="none"
              strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="9" cy="22" r="2.6" fill="white" />
            <circle cx="20" cy="20" r="2.6" fill="white" />
          </svg>
          <span className="text-lg font-bold tracking-tight">TuneFlow</span>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive
                    ? 'bg-accent/20 text-accent-soft font-semibold'
                    : 'text-zinc-400 hover:bg-white/5 hover:text-white'
                }`
              }
            >
              <span aria-hidden>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-lg bg-surface-850 p-3 text-xs text-zinc-400">
          {user ? (
            <div className="flex items-center justify-between">
              <span className="truncate font-medium text-zinc-200">{user.display_name}</span>
              <button className="text-zinc-500 hover:text-white" onClick={logout} title="Switch user">
                ⎋
              </button>
            </div>
          ) : (
            <button className="btn btn-primary w-full justify-center text-xs" onClick={openOnboarding}>
              Get started
            </button>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <nav className="flex gap-1 overflow-x-auto border-b border-white/5 bg-surface-900 px-3 py-2 md:hidden">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `whitespace-nowrap rounded-full px-3 py-1.5 text-xs ${
                  isActive ? 'bg-accent/20 text-accent-soft' : 'text-zinc-400'
                }`
              }
            >
              {item.icon} {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="min-h-0 flex-1 overflow-y-auto px-4 pb-32 pt-6 md:px-8">
          <Outlet />
        </main>
        <PlayerBar />
      </div>
    </div>
  )
}
