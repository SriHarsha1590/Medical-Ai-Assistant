import React, { useState } from "react";

const FAQAccordion = () => {
  const [openIndex, setOpenIndex] = useState(null); // Track which FAQ is open

  const faqs = [
    {
      question: "How can I donate supplies?",
      answer: "You can use the 'Donate Supplies' section to provide details about the items you wish to donate.",
    },
    {
      question: "What types of donations are most helpful?",
      answer: "Monetary donations are versatile and allow us to respond quickly. Supplies like food, clothing, and hygiene products are also highly valuable.",
    },
    {
      question: "Can I get a receipt for my donation?",
      answer: "Yes, we provide receipts for both monetary and supply donations.",
    },
  ];

  const toggleAnswer = (index) => {
    if (openIndex === index) {
      setOpenIndex(null); // If the same question is clicked, close it
    } else {
      setOpenIndex(index); // Open the clicked question
    }
  };

  return (
    <div className="accordion">
      {faqs.map((faq, index) => (
        <div key={index} className="mb-4 bg-white p-4 rounded-lg shadow-md">
          <button
            type="button"
            className="font-bold text-navy mb-2 w-full text-left"
            onClick={() => toggleAnswer(index)} // Toggle the answer visibility
          >
            {faq.question}
          </button>
          {/* Show answer only if this FAQ is open */}
          {openIndex === index && (
            <p className="text-gray mt-2">{faq.answer}</p>
          )}
        </div>
      ))}
    </div>
  );
};

export default FAQAccordion;
