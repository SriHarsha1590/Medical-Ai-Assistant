import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { ToastContainer } from 'react-toastify'; // Import ToastContainer
import 'react-toastify/dist/ReactToastify.css'; // Import the toastify CSS
import Navbar from './components/Navbar/Navbar';
import Footer from './components/Footer/Footer';
import ScrollingAlert from './components/ScrollingAlert/ScrollingAlert';
import DisasterFeed from './pages/DisasterFeed';
import ShelterPage from './pages/ShelterPage';
import { fetchAllDisasters } from './services/disasterService';
import HomeHospital from './pages/HomeHospital';
import HomeShelter from './pages/HomeShelter';
import DisasterPreparednessPage from './pages/DisasterPreparednessPage';
import AwarenessPage from './pages/AwarenessPage';
import LinksPage from './pages/LinksPage';
import Login from './pages/Login';
import SignUp from './pages/SignUp';
import HomePage from './pages/HomePage';
import VolunteerPage from './pages/VolunteerPage';
import DonatePage from './pages/Donate';

import VolunteerDashboard from './pages/VolunteerDashBoard'; // Import VolunteerDashboardPage

const CoordinatorDashboard = () => <div>Coordinator Dashboard</div>;
const AdminDashboard = () => <div>Admin Dashboard</div>;
const UserDashboard = () => <div>User Dashboard</div>;

function App() {
  const [disasters, setDisasters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { currentUser, role } = useSelector((state) => state.user);

  const [ongoingAndNew, setOngoingAndNew] = useState([]);
  const [allDisasters, setAllDisasters] = useState([]);

  const loadDisasters = async () => {
    try {
      const disasterData = await fetchAllDisasters();
      setDisasters(disasterData);
      const ongoingAndNewDisasters = disasterData.filter(disaster =>
        disaster.status === 'Ongoing' || disaster.status === 'New'
      );
      setOngoingAndNew(ongoingAndNewDisasters);
      setAllDisasters(disasterData);
    } catch (err) {
      setError('Failed to load disasters');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDisasters();
  }, []);

  const refreshDisasters = () => {
    setLoading(true);
    loadDisasters();
  };

  if (loading) {
    return <div>Loading disasters...</div>;
  }

  if (error) {
    return <div>{error}</div>;
  }

  return (
    <Router>
      <div className="flex flex-col min-h-screen">
        <Navbar />
        <ScrollingAlert alerts={ongoingAndNew} />
        <main className="flex-grow">
          <Routes>
            <Route path="/disaster-feed" element={<DisasterFeed disasters={allDisasters} refreshDisasters={refreshDisasters} />} />
            <Route path="/shelter" element={<ShelterPage />} />
            <Route path="/hospital/:hospitalId" element={<HomeHospital />} />
            <Route path="/shelter/:shelterId" element={<HomeShelter />} />
            <Route path="/preparedness/:type" element={<DisasterPreparednessPage />} />
            <Route path="/awareness/:type" element={<AwarenessPage />} />
            <Route path="/links" element={<LinksPage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<SignUp />} />
            <Route path="/donate" element={<DonatePage/>} />
            <Route path='/volunteer' element={<VolunteerPage/>} />
            <Route path='/' element={<HomePage/>} />

            {/* Dashboard route, redirects based on role */}
            <Route
              path="/dashboard"
              element={
                currentUser ? (
                  role === 'volunteer' ? <VolunteerDashboard /> :
                  role === 'coordinator' ? <CoordinatorDashboard /> :
                  role === 'admin' ? <AdminDashboard /> :
                  role === 'none' ? <UserDashboard /> :
                  <Navigate to="/" replace />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
          </Routes>
        </main>
        <Footer />
        <ToastContainer position="top-right" autoClose={5000} hideProgressBar />
      </div>
    </Router>
  );
}

export default App;
