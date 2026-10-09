import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import OnboardingModal from './components/OnboardingModal'
import ArtistPage from './pages/ArtistPage'
import DiscoverPage from './pages/DiscoverPage'
import HistoryPage from './pages/HistoryPage'
import HomePage from './pages/HomePage'
import LibraryPage from './pages/LibraryPage'
import ProfilePage from './pages/ProfilePage'
import SearchPage from './pages/SearchPage'
import SongPage from './pages/SongPage'
import { useApp } from './store/AppContext'

export default function App() {
  const { ready, onboardingOpen } = useApp()

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-zinc-500 text-sm animate-pulse">Loading TuneFlow…</div>
      </div>
    )
  }

  return (
    <>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/discover" element={<DiscoverPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/songs/:songId" element={<SongPage />} />
          <Route path="/artists/:artistId" element={<ArtistPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
      {onboardingOpen && <OnboardingModal />}
    </>
  )
}
