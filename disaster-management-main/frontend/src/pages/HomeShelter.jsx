import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { FaShare, FaArrowLeft } from "react-icons/fa";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const HomeShelter = () => {
  const { shelterId } = useParams();
  const [shelterData, setShelterData] = useState(null);
  const [error, setError] = useState(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchShelterData = async () => {
      try {
        const res = await fetch(`http://localhost:5000/api/shelters/${shelterId}`);
        if (res.ok) {
          const data = await res.json();
          setShelterData(data);
        } else {
          setError("Failed to fetch shelter data. Please try again later.");
        }
      } catch (err) {
        setError("An error occurred while fetching shelter data.");
      }
    };

    fetchShelterData();
  }, [shelterId]);

  const shareShelter = () => {
    const deepLink = `${window.location.origin}/shelter/${shelterId}`;
    if (navigator.share) {
      navigator
        .share({
          title: "Check out this shelter",
          text: "I found this shelter and thought you might be interested.",
          url: deepLink,
        })
        .catch((err) => console.error("Error sharing", err));
    } else {
      navigator.clipboard
        .writeText(deepLink)
        .then(() => toast.success("URL copied to clipboard!"))
        .catch(() => toast.error("Failed to copy URL."));
    }
  };

  const renderShelterDetails = () => (
    <div className="text-gray-700 mb-6">
      <p className="mb-4">
        <span className="font-semibold">Location:</span> {shelterData.location.address}
      </p>
      <p className="mb-4">
        <span className="font-semibold">Type:</span> {shelterData.type || "N/A"}
      </p>
      <p className="mb-4">
        <span className="font-semibold">Description:</span> {shelterData.description || "N/A"}
      </p>
      <p className="mb-4">
        <span className="font-semibold">Availability:</span>{" "}
        {shelterData.availability ? "Available" : "Unavailable"}
      </p>

      {/* Render Facilities */}
      {shelterData.facilities && shelterData.facilities.length > 0 ? (
        <div className="mb-4">
          <span className="font-semibold">Facilities:</span>
          <ul className="list-disc list-inside">
            {shelterData.facilities.map((facility, index) => (
              <li key={index}>{facility}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p className="mb-4">
          <span className="font-semibold">Facilities:</span> None
        </p>
      )}

      {/* Render Contact Info */}
      {shelterData.contact && (
        <div className="mb-4">
          <span className="font-semibold">Contact:</span>{" "}
          {shelterData.contact.email ? (
            <a href={`mailto:${shelterData.contact.email}`} className="text-teal-500 hover:underline">
              {shelterData.contact.email}
            </a>
          ) : (
            "N/A"
          )}
          , {shelterData.contact.phone || "N/A"}
        </div>
      )}
    </div>
  );

  const renderMap = () =>
    shelterData.location.coordinates ? (
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Map</h2>
        <iframe
          src={`https://www.google.com/maps?q=${shelterData.location.coordinates[1]},${shelterData.location.coordinates[0]}&z=15&output=embed`}
          width="100%"
          height="450"
          frameBorder="0"
          style={{ border: 0 }}
          allowFullScreen
          title="Shelter Map"
          className="rounded-lg shadow-md"
        ></iframe>
      </div>
    ) : (
      <p className="text-gray-600">Map information not available.</p>
    );

  const renderGallery = () => {
    const images = shelterData.photos || [];
    if (images.length === 0) return <img src={`http://localhost:5000/api/uploads/${shelterData.photo}`} alt="Shelter" className="w-full h-auto rounded-lg shadow-md mx-auto" />;

    return (
      <div className="relative">
        <div className="w-full">
          <img
            src={images[currentImageIndex]}
            alt="Shelter"
            className="w-full h-auto rounded-lg shadow-md mx-auto"
          />
        </div>
        {images.length > 1 && (
          <>
            <button
              onClick={() => setCurrentImageIndex((prev) => (prev === 0 ? images.length - 1 : prev - 1))}
              className="absolute top-1/2 left-4 transform -translate-y-1/2 text-white bg-black bg-opacity-50 p-2 rounded-full"
            >
              &#8592;
            </button>
            <button
              onClick={() => setCurrentImageIndex((prev) => (prev === images.length - 1 ? 0 : prev + 1))}
              className="absolute top-1/2 right-4 transform -translate-y-1/2 text-white bg-black bg-opacity-50 p-2 rounded-full"
            >
              &#8594;
            </button>
          </>
        )}
      </div>
    );
  };

  if (error) return <div className="text-center text-red-600 mt-10">{error}</div>;

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
              onClick={shareShelter}
            >
              <FaShare size={18} />
            </button>
          </div>
        </div>

        {shelterData ? (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h1 className="text-3xl font-bold text-gray-800 text-center mb-6">
              {shelterData.name || "Unnamed Shelter"}
            </h1>

            {/* Flexbox Layout for Image and Details */}
            <div className="flex flex-col md:flex-row items-start space-y-6 md:space-y-0 md:space-x-8">
              {/* Shelter Image Carousel */}
              <div className="w-full md:w-1/2">
                {renderGallery()}
              </div>

              {/* Shelter Details */}
              <div className="w-full md:w-1/2">
                {renderShelterDetails()}
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

export default HomeShelter;
