import React, { useEffect, useState } from 'react';
import { useUserStore } from '../store/useUserStore';
import { useAuthStore } from '../store/useAuthStore';
import { useNavigate } from 'react-router-dom';
import { Edit2, Save, X, LogOut, Camera } from 'react-feather';
import Spinner from './Spinner';
import Toast from './Toast';

const ProfilePage = () => {
  const {
    user: fetchedUser,
    loading,
    error,
    fetchProfile,
    updateProfile,
    uploadProfileImage,
    clearError,
  } = useUserStore();
  const { isAuthenticated, logout } = useAuthStore();

  const navigate = useNavigate();

  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({
    username: '',
    email: '',
  });
  const [toast, setToast] = useState({ show: false, message: '', type: 'success' });
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    if (isAuthenticated) {
      fetchProfile();
    }
  }, [fetchProfile, isAuthenticated]);

  useEffect(() => {
    if (fetchedUser) {
      setEditData({ username: fetchedUser.username, email: fetchedUser.email });
    }
  }, [fetchedUser]);

  const showToast = (message, type = 'success') => {
    setToast({ show: true, message, type });
    setTimeout(() => setToast({ show: false, message: '', type: 'success' }), 3000);
  };

  const handleEditClick = () => {
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    if (fetchedUser) {
      setEditData({ username: fetchedUser.username, email: fetchedUser.email });
    }
    clearError();
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setEditData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSaveProfile = async () => {
    try {
      await updateProfile(editData);
      setIsEditing(false);
      showToast('Profile updated successfully');
    } catch (err) {
      showToast('Failed to update profile', 'error');
    }
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Validate file type and size
    if (!file.type.startsWith('image/')) {
      showToast('Please select an image file', 'error');
      return;
    }
    
    if (file.size > 5 * 1024 * 1024) { // 5MB limit
      showToast('Image size should be less than 5MB', 'error');
      return;
    }
    
    setImageLoading(true);
    try {
      const formData = new FormData();
      formData.append('image', file);
      await uploadProfileImage(formData);
      showToast('Profile image updated successfully');
    } catch (err) {
      showToast('Failed to update profile image', 'error');
    } finally {
      setImageLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/');
    } catch (err) {
      showToast('Logout failed', 'error');
    }
  };

  if (loading && !fetchedUser) {
    return <Spinner />;
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-8">
      {toast.show && (
        <Toast 
          message={toast.message} 
          type={toast.type} 
          onClose={() => setToast({ show: false, message: '', type: 'success' })} 
        />
      )}
      
      <div className="bg-black/30 backdrop-blur-lg rounded-xl shadow-lg border border-gray-700 hover:border-purple-500 transition-all duration-300 p-6">
        <h1 className="text-3xl font-bold mb-6 text-center bg-clip-text text-transparent bg-gradient-to-r from-purple-500 to-pink-500">
          My Profile
        </h1>

        <div className="flex flex-col md:flex-row gap-8">
          {/* Profile Image Section */}
          <div className="flex flex-col items-center space-y-4">
            <div className="relative w-40 h-40 rounded-full overflow-hidden bg-gray-800 border-2 border-purple-500 shadow-lg group">
              {imageLoading ? (
                <div className="w-full h-full flex items-center justify-center bg-gray-800">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-purple-500"></div>
                </div>
              ) : fetchedUser?.profilePicUrl ? (
                <img
                  src={fetchedUser.profilePicUrl}
                  alt="Profile"
                  className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-20 h-20"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1}
                      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                    />
                  </svg>
                </div>
              )}
              
              <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 flex items-center justify-center transition-all duration-300">
                <label className="cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <div className="bg-purple-500 hover:bg-purple-600 p-2 rounded-full transition-all duration-300 transform hover:scale-110">
                    <Camera size={20} className="text-white" />
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleImageUpload}
                    disabled={imageLoading}
                  />
                </label>
              </div>
            </div>
            
            <label className="cursor-pointer">
              <span className="px-4 py-2 bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] text-sm font-medium inline-flex items-center">
                <Camera size={16} className="mr-2" />
                Change Photo
              </span>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
                disabled={imageLoading}
              />
            </label>
          </div>

          <div className="flex-1 space-y-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-white">Account Details</h2>
              {!isEditing ? (
                <button
                  onClick={handleEditClick}
                  className="bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white px-4 py-2 rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] text-sm font-medium flex items-center gap-1 transform hover:translate-y-[-2px] active:translate-y-[1px]"
                >
                  <Edit2 size={16} className="mr-1" />
                  Edit Profile
                </button>
              ) : (
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveProfile}
                    className="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white px-4 py-2 rounded-md transition-all duration-300 shadow-lg text-sm font-medium flex items-center gap-1 transform hover:translate-y-[-2px] active:translate-y-[1px]"
                  >
                    <Save size={16} className="mr-1" />
                    Save
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-500 hover:to-gray-600 text-white px-4 py-2 rounded-md transition-all duration-300 shadow-lg text-sm font-medium flex items-center gap-1 transform hover:translate-y-[-2px] active:translate-y-[1px]"
                  >
                    <X size={16} className="mr-1" />
                    Cancel
                  </button>
                </div>
              )}
            </div>

            {isEditing ? (
              <div className="space-y-4 animate-fadeIn">
                <div>
                  <label className="block text-gray-300 text-sm mb-1">Username</label>
                  <input
                    type="text"
                    name="username"
                    value={editData.username}
                    onChange={handleChange}
                    className="w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm border border-gray-700 transition-all duration-200"
                  />
                </div>
                <div>
                  <label className="block text-gray-300 text-sm mb-1">Email</label>
                  <input
                    type="email"
                    name="email"
                    value={editData.email}
                    onChange={handleChange}
                    className="w-full p-2.5 rounded-md bg-gray-800 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm border border-gray-700 transition-all duration-200"
                  />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4 bg-gray-800/50 rounded-lg p-4 border border-gray-700 hover:border-purple-500 transition-all duration-300">
                <div className="transform transition-all duration-300 hover:translate-x-1">
                  <label className="text-gray-400 text-sm">Username</label>
                  <p className="text-white text-lg font-medium">{fetchedUser?.username}</p>
                </div>
                <div className="transform transition-all duration-300 hover:translate-x-1">
                  <label className="text-gray-400 text-sm">Email</label>
                  <p className="text-white text-lg font-medium">{fetchedUser?.email}</p>
                </div>
                <div className="transform transition-all duration-300 hover:translate-x-1">
                  <label className="text-gray-400 text-sm">Member Since</label>
                  <p className="text-white text-lg font-medium">
                    {fetchedUser?.createdAt ? new Date(fetchedUser.createdAt).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    }) : ''}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-8 text-center">
          <button
            onClick={handleLogout}
            className="w-full md:w-auto bg-gradient-to-r from-[#551f2b] via-[#3a1047] to-[#1e0144] hover:from-[#6a2735] hover:via-[#4d1459] hover:to-[#2a0161] text-white px-6 py-2 rounded-md transition-all duration-300 shadow-[0_0_15px_5px_rgba(0,0,0,0.7)] text-sm font-medium flex items-center justify-center mx-auto transform hover:translate-y-[-2px] active:translate-y-[1px]"
          >
            <LogOut size={16} className="mr-2" />
            Logout
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;