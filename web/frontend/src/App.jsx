import { useState, useEffect } from 'react'
import AuthPage from './pages/AuthPage.jsx'
import Dashboard from './pages/Dashboard.jsx'

export default function App() {
  const [user, setUser] = useState(() => {
    const u = localStorage.getItem('username')
    const t = localStorage.getItem('token')
    return (u && t) ? { username: u, token: t } : null
  })

  function handleLogin(username, token) {
    localStorage.setItem('username', username)
    localStorage.setItem('token', token)
    setUser({ username, token })
  }

  function handleLogout() {
    localStorage.removeItem('username')
    localStorage.removeItem('token')
    setUser(null)
  }

  if (!user) {
    return <AuthPage onLogin={handleLogin} />
  }

  return <Dashboard user={user} onLogout={handleLogout} />
}
