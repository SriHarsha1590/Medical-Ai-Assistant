import { useEffect, useState } from 'react'; 
import { useParams } from 'react-router-dom';
import axios from 'axios';
import Sidebar from '../components/PreparednessPage/Sidebar';
import { ClipLoader } from 'react-spinners'; // Add spinner for loading state

const DisasterPreparednessPage = () => {
  const { type } = useParams();
  const disasterType = type || 'Earthquake';

  const [preparednessData, setPreparednessData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setError(null); // Clear any previous error
    setPreparednessData(null); // Reset data while new data loads
    const fetchData = async () => {
      try {
        const response = await axios.get(`/api/preparedness/${disasterType}`);
        setPreparednessData(response.data);
      } catch (err) {
        setError('Error fetching disaster preparedness data');
        console.error(err); // Log the actual error for debugging
      }
    };
    fetchData();
  }, [type]);

  if (error) {
    return <div className="text-center text-red-500">{error}</div>;
  }

  if (!preparednessData) {
    return (
      <div className="text-center text-gray-500">
        <ClipLoader size={50} color="#4ade80" /> Loading...
      </div>
    );
  }

  return (
    <div className="flex min-h-screen font-sans bg-gray-50">
      {/* Sidebar */}
      <Sidebar className="bg-navy text-white p-6" />

      {/* Main Content */}
      <div className="flex-1 p-10 bg-white flex flex-col items-center space-y-8 md:space-y-12">
        {/* Heading */}
        <h1 className="text-4xl font-semibold text-navy mb-6">{preparednessData.type} Preparedness</h1>

        {/* Main Content Area */}
        <div className="flex flex-col md:flex-row w-full max-w-6xl space-y-6 md:space-y-0 md:space-x-8">
          {/* Left Column: Text Content */}
          <div className="flex-1 space-y-6">
            {/* Description Section */}
            <section className="p-6 rounded-lg bg-lightGray shadow-lg">
              <h2 className="text-2xl font-medium text-teal mb-4">Description</h2>
              <p className="text-gray-700 text-lg">{preparednessData.description}</p>
            </section>

            {/* Guidelines Section */}
            <section className="p-6 rounded-lg bg-lightGray shadow-lg">
              <h2 className="text-2xl font-medium text-teal mb-4">Guidelines</h2>
              <div className="space-y-6">
                <div>
                  <h3 className="font-semibold text-navy text-lg mb-2">Before</h3>
                  <ul className="list-disc pl-5 space-y-2 text-gray-700">
                    {preparednessData.guidelines.before.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold text-navy text-lg mb-2">During</h3>
                  <ul className="list-disc pl-5 space-y-2 text-gray-700">
                    {preparednessData.guidelines.during.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold text-navy text-lg mb-2">After</h3>
                  <ul className="list-disc pl-5 space-y-2 text-gray-700">
                    {preparednessData.guidelines.after.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </section>

            {/* Key Preparation Steps Section */}
            <section className="p-6 rounded-lg bg-lightGray shadow-lg">
              <h2 className="text-2xl font-medium text-teal mb-4">Key Preparation Steps</h2>
              <ul className="list-decimal pl-5 space-y-2 text-gray-700">
                {preparednessData.steps.map((step, index) => (
                  <li key={index}>{step}</li>
                ))}
              </ul>
            </section>
          </div>

          {/* Right Column: Media (Image and Video) */}
          <div className="md:w-1/3 flex flex-col items-center space-y-6">
            {preparednessData.media && preparednessData.media.map((mediaItem, index) => (
              <div key={index} className="w-full p-4 rounded-lg shadow-lg">
                {mediaItem.type === 'image' ? (
                  <img
                    src={mediaItem.url}
                    alt="Disaster preparedness"
                    className="w-full h-auto rounded-lg"
                  />
                ) : mediaItem.type === 'video' ? (
                  <iframe
                    width="100%"
                    height="315"
                    src={mediaItem.url}
                    title="Disaster Preparedness Video"
                    className="rounded-lg"
                    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  ></iframe>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DisasterPreparednessPage;
