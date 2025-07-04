import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { FaShare, FaArrowLeft } from "react-icons/fa";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const HomeHospital = () => {
  const { hospitalId } = useParams();
  const [hospitalData, setHospitalData] = useState(null);
  const [error, setError] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0); // State to manage the carousel index
  const navigate = useNavigate();

  useEffect(() => {
    const fetchHospitalData = async () => {
      try {
        const res = await fetch(`http://localhost:5000/api/hospitals/${hospitalId}`);
        if (res.ok) {
          const data = await res.json();
          setHospitalData(data);
        } else {
          setError("Failed to fetch hospital data");
        }
      } catch (error) {
        setError("Error fetching hospital data");
      }
    };

    fetchHospitalData();
  }, [hospitalId]);

  const shareHospital = () => {
    const deepLink = `${window.location.origin}/hospital/${hospitalId}`;
    if (navigator.share) {
      navigator.share({
        title: "Check out this hospital",
        text: "I found this hospital and thought you might be interested.",
        url: deepLink,
      });
    } else {
      alert(`Shareable URL: ${deepLink}`);
    }
  };

  const renderMap = () =>
    hospitalData?.location?.coordinates ? (
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Map</h2>
        <iframe
          src={`https://www.google.com/maps?q=${hospitalData.location.coordinates[1]},${hospitalData.location.coordinates[0]}&z=15&output=embed`}
          width="100%"
          height="450"
          frameBorder="0"
          style={{ border: 0 }}
          allowFullScreen
          title="Hospital Map"
          className="rounded-lg shadow-md"
        ></iframe>
      </div>
    ) : (
      <p className="text-gray-600">Map information not available.</p>
    );

  // Carousel Handlers
  const handleNextImage = () => {
    if (hospitalData && hospitalData.photos && hospitalData.photos.length > 0) {
      setCurrentImageIndex((prevIndex) =>
        prevIndex === hospitalData.photos.length - 1 ? 0 : prevIndex + 1
      );
    }
  };

  const handlePrevImage = () => {
    if (hospitalData && hospitalData.photos && hospitalData.photos.length > 0) {
      setCurrentImageIndex((prevIndex) =>
        prevIndex === 0 ? hospitalData.photos.length - 1 : prevIndex - 1
      );
    }
  };

  // Render Image Carousel (with first image as default)
  const renderImage = () => {
    const images = hospitalData?.photos || [];
    const currentImage =
      images.length > 0 ? images[currentImageIndex] : hospitalData?.photo;

    return (
      <div className="relative">
        <img
          src={
            currentImage
              ? `http://localhost:5000/api/uploads/${currentImage}`
              : "https://via.placeholder.com/150"
          }
          alt="Hospital"
          className="w-full md:w-2/3 h-auto rounded-lg shadow-md mx-auto"
        />
        {images.length > 1 && (
          <div className="absolute top-1/2 left-0 transform -translate-y-1/2 text-white">
            <button
              onClick={handlePrevImage}
              className="bg-black bg-opacity-50 p-2 rounded-full"
            >
              &#8249;
            </button>
          </div>
        )}
        {images.length > 1 && (
          <div className="absolute top-1/2 right-0 transform -translate-y-1/2 text-white">
            <button
              onClick={handleNextImage}
              className="bg-black bg-opacity-50 p-2 rounded-full"
            >
              &#8250;
            </button>
          </div>
        )}
      </div>
    );
  };

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div className="bg-gray-100 min-h-screen">
      <ToastContainer />
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-between mb-4">
          <button
            className="flex items-center text-gray-600 hover:text-gray-800"
            onClick={() => navigate("/shelter")}
          >
            <FaArrowLeft className="mr-2" />
            <span>Back</span>
          </button>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">Share</span>
            <button
              className="text-gray-600 hover:text-gray-800"
              onClick={shareHospital}
            >
              <FaShare size={18} />
            </button>
          </div>
        </div>

        {hospitalData ? (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h1 className="text-3xl font-bold text-gray-800 text-center mb-6">
              {hospitalData.name || "Unnamed Hospital"}
            </h1>

            {/* Flexbox Layout for Image and Details */}
            <div className="flex flex-col md:flex-row items-start space-y-6 md:space-y-0 md:space-x-8">
              {/* Render Image Carousel */}
              <div className="w-full md:w-1/2">{renderImage()}</div>

              {/* Hospital Details */}
              <div className="w-full md:w-1/2 text-gray-700">
                <p className="mb-4">
                  <span className="font-semibold">Address:</span> {hospitalData.location.address}
                </p>
                <p className="mb-4">
                  <span className="font-semibold">Facilities:</span> {hospitalData.facilities.join(", ")}
                </p>
                <p className="mb-4">
                  <span className="font-semibold">Capacity:</span> {hospitalData.capacity}
                </p>
                <p className="mb-4">
                  <span className="font-semibold">Contact Number:</span> {hospitalData.contact.phone}
                </p>
                <p className="mb-4">
                  <span className="font-semibold">Email:</span>{" "}
                  <a
                    href={`mailto:${hospitalData.contact.email}`}
                    className="text-teal-500 hover:underline"
                  >
                    {hospitalData.contact.email}
                  </a>
                </p>
              </div>
            </div>

            {renderMap()}
          </div>
        ) : (
          <p className="text-center text-gray-600">Loading...</p>
        )}
      </div>
    </div>
  );
};

export default HomeHospital;
