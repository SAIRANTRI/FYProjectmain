import React, { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../store/useAuthStore';
import { Eye, EyeOff } from 'react-feather';
import Spinner from './Spinner';

export default function LoginPage() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [formErrors, setFormErrors] = useState({});
  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState('');

  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/upload";

  const { login, isLoading, error, clearError } = useAuthStore();

  const validateForm = () => {
    const errors = {};
    if (!formData.email) errors.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(formData.email)) errors.email = "Email is invalid";
    
    if (!formData.password) errors.password = "Password is required";
    else if (formData.password.length < 6) errors.password = "Password must be at least 6 characters";
    
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    
    // Clear error for this field when user starts typing
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: null }));
    }
    if (error) clearError();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    try {
      await login({
        email: formData.email,
        password: formData.password
      });
      setToastMessage('Login successful!');
      setShowToast(true);
      navigate(from); 
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="w-full flex justify-center mb-10">
      <div className="w-full max-w-sm px-6 py-8 bg-black/30 backdrop-blur-lg rounded-xl shadow-lg border border-gray-700 hover:border-purple-500 transition-all duration-300 transform hover:scale-[1.01]">
        <h2 className="text-2xl font-bold text-white mb-5 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500">
          Login to Albumify
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="email" className="text-gray-300 block mb-1 text-sm">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              className={`w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm transition-all duration-200 ${
                formErrors.email ? 'border border-red-500 focus:ring-red-500' : 'border border-gray-700'
              }`}
              placeholder="your@email.com"
            />
            {formErrors.email && (
              <p className="mt-1 text-sm text-red-500 transition-all duration-200">{formErrors.email}</p>
            )}
          </div>

          <div>
            <label htmlFor="password" className="text-gray-300 block mb-1 text-sm">Password</label>
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                id="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className={`w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm transition-all duration-200 ${
                  formErrors.password ? 'border border-red-500 focus:ring-red-500' : 'border border-gray-700'
                }`}
                placeholder="••••••••"
              />
              <button
                type="button"
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-white transition-colors"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            {formErrors.password && (
              <p className="mt-1 text-sm text-red-500 transition-all duration-200">{formErrors.password}</p>
            )}
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
              <a href="#" className="text-purple-400 hover:underline transition-all duration-200">
                Forgot password?
              </a>
            </div>
          </div>

          {error && (
            <div className="p-2 text-sm text-red-500 bg-red-500/10 border border-red-500/20 rounded-md transition-all duration-300">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white text-base py-2.5 rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] transform hover:translate-y-[-2px] active:translate-y-[1px] disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="flex justify-center items-center">
                <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white mr-2"></div>
                <span>Logging in...</span>
              </div>
            ) : (
              "Log In"
            )}
          </button>
        </form>

        <p className="mt-5 text-center text-sm text-gray-400">
          Don't have an account?{" "}
          <a href="/signup" className="text-purple-400 hover:underline transition-all duration-200">
            Sign up
          </a>
        </p>
      </div>
    </div>
  )
}