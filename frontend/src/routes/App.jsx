import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { useUserStore } from '../store/useUserStore';
import Header from '../components/Header';
import Footer from '../components/Footer';
import SplashImage from '../assets/Splash4Edddc9Ajpg.jpeg';

function App() {
  const { fetchProfile } = useUserStore();

  // Fetch the user profile when the app initializes
  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  return (
    <div
      style={{
        backgroundImage: `url(${SplashImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}
    >
      {/* Header stays at the top */}
      <div className="w-full flex justify-center px-5">
        <Header />
      </div>

      {/* Scrollable Main Content */}
      <div className="w-full flex-grow flex justify-center overflow-auto px-5">
        <Outlet />
      </div>

      {/* Footer stays at the bottom */}
      <div className="w-full flex justify-center px-5">
        <Footer />
      </div>
    </div>
  );
}

export default App;