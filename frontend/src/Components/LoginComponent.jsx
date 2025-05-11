import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export default function LoginPage() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
  });

  // Initialize useNavigate outside the conditional block
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // In a real app, you would handle authentication here
    navigate("/dashboard");
  };

  return (
    <div className="w-full flex justify-center mb-10">
      <div className="w-full max-w-sm px-6 py-8 bg-black/30 backdrop-blur-lg rounded-xl shadow-lg border border-gray-700 hover:border-purple-500 transition-all duration-300">
        <h2 className="text-2xl font-bold text-white mb-5 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500">
          Login to Albumify
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="text-gray-300 block mb-1 text-sm">Username</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              required
            />
          </div>

          <div>
            <label htmlFor="email" className="text-gray-300 block mb-1 text-sm">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className="w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              placeholder="your@email.com"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="text-gray-300 block mb-1 text-sm">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm"
              placeholder="••••••••"
              required
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <input
                id="remember-me"
                name="remember-me"
                type="checkbox"
                className="h-4 w-4 rounded border-gray-600 bg-gray-700 text-purple-600 focus:ring-purple-500"
              />
              <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-300">
                Remember me
              </label>
            </div>
            <div className="text-sm">
              <a href="#" className="text-purple-400 hover:underline">
                Forgot password?
              </a>
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white text-base py-2.5 rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)]"
          >
            Log In
          </button>
        </form>

        <p className="mt-5 text-center text-sm text-gray-400">
          Don't have an account?{" "}
          <a href="/signup" className="text-purple-400 hover:underline">
            Sign up
          </a>
        </p>
      </div>
    </div>
  )
}