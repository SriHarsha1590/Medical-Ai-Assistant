import React, { useState, useEffect, useRef } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useSelector, useDispatch } from "react-redux";
import { FaUserCircle, FaBars, FaChevronDown, FaChevronUp } from "react-icons/fa";
import { signoutSuccess } from "../../redux/userSlice";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const { currentUser, role } = useSelector((state) => state.user);

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [openDropdown, setOpenDropdown] = useState(null);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const dropdownRef = useRef(null);

  const navLinks = [
    { to: "/", label: "Home" },
    { to: "/disaster-feed", label: "Disaster Feed" },
    { to: "/shelter", label: "Shelters" },
  ];

  const dropdownLinks = [
    {
      label: "Get Involved",
      subLinks: [
        { to: "/volunteer", label: "Volunteer" },
        { to: "/donate", label: "Donate" },
      ],
    },
    {
      label: "Resources",
      subLinks: [
        { to: "/preparedness/Earthquake", label: "Preparedness" },
        { to: "/awareness/Earthquake", label: "Awareness" },
        { to: "/links", label: "Important links" },
      ],
    },
  ];

  const dashboardLinks = {
    none: { to: "/dashboard", label: "User Dashboard" },
    volunteer: { to: "/dashboard/volunteer", label: "Volunteer Dashboard" },
    coordinator: { to: "/dashboard/coordinator", label: "Coordinator Dashboard" },
    admin: { to: "/dashboard/admin", label: "Admin Dashboard" },
  };

  const getLinkClassName = (path) => {
    return location.pathname === path
      ? "text-teal font-bold"
      : "text-gray hover:text-teal transition-colors duration-300";
  };

  const toggleDropdown = (index) => {
    setOpenDropdown(openDropdown === index ? null : index);
  };

  const toggleUserMenu = () => {
    setIsUserMenuOpen((prev) => !prev);
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    dispatch(signoutSuccess());
    navigate("/login"); // Redirect to login page after logout
  };

  const handleDashboardClick = (e) => {
    e.stopPropagation(); // Prevent closing the user menu
    navigate(dashboardLinks[role]?.to || "/dashboard"); // Navigate to the user's dashboard
  };

  const closeDropdowns = (event) => {
    if (
      dropdownRef.current &&
      event?.target && 
      !dropdownRef.current.contains(event.target) && 
      !event.target.closest(".user-menu") && 
      !event.target.closest(".user-icon")
    ) {
      setOpenDropdown(null);
      setIsMenuOpen(false);
      setIsUserMenuOpen(false);
    }
  };
  
  useEffect(() => {
    const handleClickOutside = (event) => {
      closeDropdowns(event);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);
  

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setOpenDropdown(null);
        setIsMenuOpen(false);
        setIsUserMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);


  return (
    <nav className="bg-navy-dark p-4 text-white">
      <div className="container mx-auto flex justify-between items-center">
       
        <div className="text-lg font-bold">
          <Link to="/">Disaster Management</Link>
        </div>

        {/* Hamburger Icon for Mobile */}
        <div
          className="lg:hidden cursor-pointer"
          onClick={() => setIsMenuOpen(!isMenuOpen)}
        >
          <FaBars className="text-2xl" />
        </div>

        {/* Navbar Links for Desktop and Mobile */}
        <div
          className={`${isMenuOpen ? "block" : "hidden"
            } lg:flex lg:space-x-6 mt-4 lg:mt-0 w-full lg:w-auto flex-col lg:flex-row items-center`}
          ref={dropdownRef}
        >
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`${getLinkClassName(link.to)} block lg:inline-block py-2 px-4`}
            >
              {link.label}
            </Link>
          ))}

          {/* Dropdown Menus */}
          {dropdownLinks.map((dropdown, index) => (
            <div
              key={index}
              className="relative lg:inline-flex items-center py-2 px-4 w-full lg:w-auto"
            >
              <button
                className="text-gray hover:text-teal w-full lg:w-auto flex justify-between items-center"
                onClick={() => toggleDropdown(index)}
              >
                {dropdown.label}
                <span className="ml-2">
                  {openDropdown === index ? <FaChevronUp /> : <FaChevronDown />}
                </span>
              </button>

              <div
                className={`${openDropdown === index ? "block" : "hidden"
                  } bg-navy text-white p-2 space-y-2 rounded-md z-10 lg:absolute lg:left-0 lg:top-full transition duration-200 ease-in-out`}
              >
                {dropdown.subLinks.map((subLink) => (
                  <Link
                    key={subLink.to}
                    to={subLink.to}
                    className="block hover:text-teal transition-colors duration-300"
                  >
                    {subLink.label}
                  </Link>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* User Icon or Login/Signup */}
        <div className="relative">
          {currentUser ? (
            <div className="flex items-center space-x-2">
              <FaUserCircle
                className="user-icon text-white text-3xl cursor-pointer"
                onClick={(e) => {
                  e.stopPropagation(); // Prevent unintended menu toggle
                  toggleUserMenu();
                }}
              />
              {isUserMenuOpen && (
                <div
                  className="user-menu absolute top-full mt-2 right-0 w-48 bg-white text-navy rounded shadow-lg z-50"
                  onClick={(e) => e.stopPropagation()} // Prevent dropdown closure on click
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent menu closure
                      handleDashboardClick(e);
                    }}
                    className="block px-4 py-2 hover:bg-gray-light transition duration-200 w-full text-left"
                  >
                    {dashboardLinks[role]?.label || "Dashboard"}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent menu closure
                      handleLogout();
                    }}
                    className="block w-full text-left px-4 py-2 hover:bg-gray-light transition duration-200"
                  >
                    Logout
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="flex space-x-4">
              <Link
                to="/login"
                className="text-gray hover:text-teal transition duration-300"
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="text-gray hover:text-teal transition duration-300"
              >
                Signup
              </Link>
            </div>
          )}
        </div>

      </div>
    </nav>
  );
};

export default Navbar;
