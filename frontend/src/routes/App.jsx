import { useEffect, Suspense, lazy } from 'react';
import { Outlet, Route, Routes, Navigate, useLocation } from 'react-router-dom';
import { useUserStore } from '../store/useUserStore';
import { useAuthStore } from '../store/useAuthStore';
import Header from '../components/Header';
import Footer from '../components/Footer';
import Spinner from '../components/Spinner';
import ProtectedRoute from '../components/ProtectedRoute';
import SplashImage from '../assets/Splash4Edddc9Ajpg.jpeg';

// Lazy load components for better performance
const Home = lazy(() => import('./Home'));
const Login = lazy(() => import('./Login'));
const Signup = lazy(() => import('./Signup'));
const Upload = lazy(() => import('./Upload'));
const Profile = lazy(() => import('./Profile'));

// Page transition component
const PageTransition = ({ children }) => {
  const location = useLocation();
  
  return (
    <div 
      key={location.pathname}
      className="animate-fadeIn w-full flex-grow flex justify-center overflow-auto px-5"
    >
      {children}
    </div>
  );
};

function AppLayout() {
  const { fetchProfile } = useUserStore();
  const { checkAuth, isCheckingAuth } = useAuthStore();

  // Fetch the user profile when the app initializes
  useEffect(() => {
    checkAuth();
    fetchProfile();
  }, [checkAuth, fetchProfile]);

  return (
    <div
      style={{
        backgroundImage: `url(${SplashImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundAttachment: 'fixed',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}
    >
      {/* Header stays at the top */}
      <div className="w-full flex justify-center px-5 top-0 z-10">
        <Header />
      </div>

      {/* Scrollable Main Content */}
      <Suspense fallback={
        <div className="w-full flex-grow flex justify-center items-center">
          <Spinner />
        </div>
      }>
        <PageTransition>
          <Outlet />
        </PageTransition>
      </Suspense>

      {/* Footer stays at the bottom */}
      <div className="w-full flex justify-center px-5">
        <Footer />
      </div>
    </div>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<Home />} />
        <Route path="login" element={<Login />} />
        <Route path="signup" element={<Signup />} />
        <Route path="upload" element={
          <ProtectedRoute>
            <Upload />
          </ProtectedRoute>
        } />
        <Route path="profile" element={
          <ProtectedRoute>
            <Profile />
          </ProtectedRoute>
        } />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

export default App;