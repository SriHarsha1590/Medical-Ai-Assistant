import React, { useState } from "react";
import Donation from '../components/Donate/Donation';
import DonateSupplies from "../components/Donate/DonateSupplies";
import FAQAccordion from "../components/Donate/FAQAccordion";

const DonatePage = () => {
  // Shared state for tracking donation-related updates
  const [donationInfo, setDonationInfo] = useState(null);
  const [suppliesInfo, setSuppliesInfo] = useState(null);

  // Callback for Donation updates
  const handleDonationUpdate = (data) => {
    setDonationInfo(data); // Store donation info
  };

  // Callback for Supplies updates
  const handleSuppliesUpdate = (data) => {
    setSuppliesInfo(data); // Store supplies donation info
  };

  return (
    <div className="get-involved-page bg-gray-light min-h-screen p-8">
      {/* Header Section */}
      <header className="text-center mb-12">
        <h1 className="text-3xl font-bold text-navy mb-4">Make a Difference Today</h1>
        <p className="text-gray text-lg mb-6">
          Your generosity can change lives. Whether through monetary donations or supplies,
          every contribution brings us closer to helping those in need.
        </p>

        {/* Why Donate Section */}
        <section className="why-donate-section bg-teal-light p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold text-navy mb-4">Why Your Donation Matters</h2>
          <ul className="list-disc list-inside text-gray text-left">
            <li><strong>₹500</strong> provides nutritious meals for a family in need.</li>
            <li><strong>₹2500</strong> offers critical supplies for disaster relief efforts.</li>
            <li><strong>₹5000</strong> helps rebuild homes and offers shelter to the displaced.</li>
            <li>Or donate essential <strong>supplies</strong> like clothing, food, or medical kits directly to those in need.</li>
          </ul>
        </section>
      </header>



      {/* Main Content */}
      <main className="space-y-16">
        {/* Donate Money Section */}
        <section className="donation-section">
          <h2 className="text-2xl font-bold text-navy mb-6 text-center">
            Donate Money
          </h2>
          <Donation onDonationUpdate={handleDonationUpdate} />
        </section>

        {/* Donate Supplies Section */}
        <section className="supplies-section">
          <h2 className="text-2xl font-bold text-navy mb-6 text-center">
            Donate Supplies
          </h2>
          <DonateSupplies onSuppliesUpdate={handleSuppliesUpdate} />
        </section>



        {/* FAQ Section */}
        <section className="faq-section">
          <h2 className="text-2xl font-bold text-navy mb-6 text-center">FAQs</h2>
          <FAQAccordion />
        </section>

        {/* Real-Time Info Display */}
        <section className="info-display">
          {donationInfo && (
            <div className="bg-teal-light p-4 rounded-lg shadow-md mb-6">
              <h3 className="text-navy font-bold text-xl mb-2">Donation Info</h3>
              <pre className="text-gray">{JSON.stringify(donationInfo, null, 2)}</pre>
            </div>
          )}
          {suppliesInfo && (
            <div className="bg-teal-light p-4 rounded-lg shadow-md">
              <h3 className="text-navy font-bold text-xl mb-2">
                Supplies Donation Info
              </h3>
              <pre className="text-gray">{JSON.stringify(suppliesInfo, null, 2)}</pre>
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default DonatePage;
