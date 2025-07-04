import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import AwarenessSidebar from '../components/DisasterAwareness/AwarenessSidebar';
import ClipLoader from 'react-spinners/ClipLoader'; // Add clip loader (from react-spinners)

const AwarenessPage = () => {
  const { type } = useParams();
  const [awarenessData, setAwarenessData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true); // Set initial loading state

  useEffect(() => {
    const fetchAwarenessData = async () => {
      setLoading(true); // Set loading to true when starting the request
      setError(null); // Reset the error state
      setAwarenessData(null); // Reset the awareness data
      try {
        const response = await axios.get(`/api/awareness/${type}`);
        setAwarenessData(response.data); // Set data after successful request
      } catch (err) {
        setError('Error fetching awareness data'); // Handle errors
      } finally {
        setLoading(false); // Set loading to false when request completes
      }
    };

    fetchAwarenessData();
  }, [type]); // Re-run the fetch when the 'type' changes

  // Error handling
  if (error) {
    return <div className="text-center text-red-500 mt-10">{error}</div>;
  }

  // Loading state with clip loader
  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <ClipLoader color="#36D7B7" loading={loading} size={50} />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen font-sans bg-gray-100">
      {/* Sidebar */}
      <AwarenessSidebar />

      {/* Main Content */}
      <div className="flex-1 p-10 bg-white rounded-lg shadow-lg m-4">
        <h1 className="text-4xl font-semibold text-navy mb-8">{awarenessData.type} Awareness</h1>
        <p className="text-lg text-gray-700 mb-8">{awarenessData.introduction}</p>

        {/* Key Awareness Topics */}
        <div className="space-y-8">
          {awarenessData.topics.map((topic, index) => (
            <section key={index} className="p-6 rounded-lg bg-lightGray shadow-md">
              <h2 className="text-2xl font-medium text-teal mb-2">{topic.title}</h2>
              <ul className="list-disc pl-5 space-y-2 text-gray-700">
                {topic.description.map((point, idx) => ( // Corrected from 'points' to 'description'
                  <li key={idx}>{point}</li>
                ))}
              </ul>
            </section>
          ))}
        </div>

        {/* Visual Media */}
{/* Visual Media */}
<div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
  {awarenessData.media.map((mediaItem, index) => (
    <div key={index} className="w-full rounded-lg  overflow-hidden">
      
      {mediaItem.type === 'image' ? (
        <div className='shadow-lg'>
        <img
          src={mediaItem.url}
          alt="Awareness content"
          className="w-full h-auto rounded-lg"
        />
        </div>
      ) : (
        <div className="relative w-full h-0 pb-[56.25%]"> {/* Aspect ratio 16:9 */}
          <iframe
            width="100%"
            height="100%"
            src={mediaItem.url}
            title="Awareness video"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            className="absolute top-0 left-0 w-full h-full rounded-lg"
          ></iframe>
        </div>
      )}
    </div>
  ))}
</div>


        {/* Resources Section */}
        <div className="mt-10">
          <h2 className="text-2xl font-semibold text-teal mb-4">Resources & Further Reading</h2>
          <ul className="list-disc pl-5 space-y-2 text-gray-700">
            {awarenessData.resources.map((resource, index) => (
              <li key={index}>
                <a href={resource.link} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
                  {resource.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AwarenessPage;
