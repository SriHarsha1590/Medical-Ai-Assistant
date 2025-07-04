// components/AwarenessSidebar.js
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const disasterTypes = [
  { name: 'Earthquake', path: '/awareness/Earthquake' },
  { name: 'Flood', path: '/awareness/Flood' },
  { name: 'Fire', path: '/awareness/Fire' },
  { name: 'Cyclone', path: '/awareness/Cyclone' },
];

const AwarenessSidebar = () => {
  const [isOpen, setIsOpen] = useState(true);
  const location = useLocation();

  const toggleSidebar = () => setIsOpen(!isOpen);

  return (
    <div className={`flex flex-col h-full ${isOpen ? 'w-64' : 'w-16'} bg-navy text-white rounded-r-lg p-4 mt-20 ml-4 transition-all duration-300 shadow-lg`}>
      {/* Toggle Button for Mobile */}
      <button onClick={toggleSidebar} className="lg:hidden mb-4 p-2 text-white bg-teal rounded-md transition duration-200 hover:bg-teal-dark">
        {isOpen ? '<<' : '>>'}
      </button>

      {/* Sidebar Title */}
      <h2 className={`text-2xl font-bold mb-6 ${isOpen ? 'block' : 'hidden'}`}>
        Awareness Topics
      </h2>

      {/* Disaster Types List */}
      <ul className="space-y-4">
        {disasterTypes.map((disaster) => (
          <li key={disaster.name}>
            <Link 
              to={disaster.path}
              className={`flex items-center space-x-2 p-3 rounded-md transition duration-200 ${
                location.pathname === disaster.path 
                  ? 'bg-teal text-navy font-semibold shadow-md' 
                  : 'hover:bg-lightGray hover:text-navy'
              }`}
            >
              <span>{disaster.name}</span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default AwarenessSidebar;
