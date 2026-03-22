import { useState } from 'react'
import { api } from '../api.js'

export default function AuthPage({ onLogin }) {
  const [mode, setMode]         = useState('login')   // 'login' | 'register'
  const [form, setForm]         = useState({ username: '', password: '', confirm: '' })
  const [error, setError]       = useState('')
  const [success, setSuccess]   = useState('')
  const [loading, setLoading]   = useState(false)

  function set(field, value) {
    setForm(f => ({ ...f, [field]: value }))
    setError('')
    setSuccess('')
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setSuccess('')

    if (!form.username.trim()) return setError('用户名不能为空')
    if (!form.password)        return setError('密码不能为空')

    if (mode === 'register') {
      if (!form.confirm)                   return setError('请确认密码')
      if (form.password !== form.confirm)  return setError('两次密码不一致')
    }

    setLoading(true)
    try {
      if (mode === 'register') {
        await api.register(form.username.trim(), form.password)
        setSuccess('注册成功，请登录')
        setMode('login')
        setForm(f => ({ ...f, confirm: '', password: '' }))
      } else {
        const data = await api.login(form.username.trim(), form.password)
        onLogin(data.username, data.access_token)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const isLogin = mode === 'login'

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo / Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-indigo-600 text-white text-3xl mb-4 shadow-lg">
            🎬
          </div>
          <h1 className="text-3xl font-bold text-gray-900">NeRF SR Studio</h1>
          <p className="text-gray-500 mt-1 text-sm">超分辨率 · 三维重建一体化平台</p>
        </div>

        {/* Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-6">
            {isLogin ? '登录账号' : '创建账号'}
          </h2>

          {/* Success banner */}
          {success && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-green-50 border border-green-200 text-green-700 text-sm">
              ✅ {success}
            </div>
          )}

          {/* Error banner */}
          {error && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
              ⚠️ {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                用户名 <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={form.username}
                onChange={e => set('username', e.target.value)}
                placeholder="请输入用户名"
                autoFocus
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                密码 <span className="text-red-400">*</span>
              </label>
              <input
                type="password"
                value={form.password}
                onChange={e => set('password', e.target.value)}
                placeholder="请输入密码"
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
              />
              {!isLogin && (
                <p className="mt-1 text-xs text-gray-400">建议使用字母+数字组合以提高安全性</p>
              )}
            </div>

            {/* Confirm password (register only) */}
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  确认密码 <span className="text-red-400">*</span>
                </label>
                <input
                  type="password"
                  value={form.confirm}
                  onChange={e => set('confirm', e.target.value)}
                  placeholder="再次输入密码"
                  className="w-full px-4 py-2.5 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition"
                />
              </div>
            )}

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-medium rounded-lg text-sm transition focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
            >
              {loading ? '处理中…' : (isLogin ? '登录' : '注册')}
            </button>
          </form>

          {/* Toggle link */}
          <div className="mt-5 text-center text-sm text-gray-500">
            {isLogin ? (
              <>还没有账号？{' '}
                <button
                  onClick={() => { setMode('register'); setError(''); setSuccess('') }}
                  className="text-indigo-600 hover:underline font-medium"
                >注册</button>
              </>
            ) : (
              <>已有账号？{' '}
                <button
                  onClick={() => { setMode('login'); setError(''); setSuccess('') }}
                  className="text-indigo-600 hover:underline font-medium"
                >返回登录</button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
