// LinksPage.js

import React from 'react';

const LinksPage = () => {
  const linkGroups = [
    {
      title: 'Alerts Hub',
      links: [
        { name: 'Cyclone', url: 'https://mausam.imd.gov.in/imd_latest/contents/cyclone.php' },
        { name: 'Rainfall', url: 'https://mausam.imd.gov.in/' },
        { name: 'Flood', url: 'https://mausam.imd.gov.in/imd_latest/contents/flash_flood.php' },
        { name: 'Earthquake', url: 'https://riseq.seismo.gov.in/riseq/earthquake' },
        { name: 'Tsunami', url: 'https://incois.gov.in/' },
        { name: 'Landslide', url: 'https://www.gsi.gov.in/webcenter/portal/OCBIS/pageGeoInfo/pageLANDSLIDEHAZRD?_adf.ctrl-state=osayshkkp_63' },
        { name: 'Avalanche', url: 'https://drdo.gov.in/drdo/avalanche-warning-bulletin' },
        { name: 'Drought', url: 'https://agriwelfare.gov.in/' },
        { name: 'Thunderstorm', url: 'https://srf.tropmet.res.in/srf/ts_prediction_system/index.php' },
        { name: 'Wildfire', url: 'https://fsiforestfire.gov.in/index.php' },
      ],
    },
    {
      title: 'Sector-Specific Forecasts',
      links: [
        { name: 'Air Quality', url: 'https://ews.tropmet.res.in/' },
        { name: 'Marine Forecast', url: 'https://mausam.imd.gov.in/imd_latest/contents/index_fisherman.php' },
        { name: 'Weather Forecast', url: 'https://mausam.imd.gov.in/imd_latest/contents/all_india_forcast_bulletin.php' },
        { name: 'Tourism Forecast', url: 'https://internal.imd.gov.in/pages/tourism_forecast_mausam.php' },
        { name: 'Fog', url: 'https://mausam.imd.gov.in/' },
        { name: 'Air Quality (SILAM)', url: 'https://nwp.imd.gov.in/silam_imd.php' },
        { name: 'Pilgrimage Forecast', url: 'https://mausam.imd.gov.in/imd_latest/contents/pilgrim_forecast.php' },
        { name: 'Highway Forecast', url: 'https://internal.imd.gov.in/pages/highway_forecast_mausam.php' },
      ],
    },
    {
      title: 'Other Important Links',
      links: [
        { name: 'NDMA', url: 'https://ndma.gov.in/' },
        { name: 'NDRF', url: 'https://www.ndrf.gov.in/' },
        { name: 'SDRF', url: 'https://ndma.gov.in/Response/SDRF' },
      ],
    },
  ];

  return (
    <div className="bg-lightGray p-8 font-sans">
      {linkGroups.map((group, index) => (
        <div key={index} className="mb-8 p-6 bg-white rounded-lg shadow-md">
          <h2 className="text-navy mb-4 font-bold text-xl">{group.title}</h2>
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="text-left text-gray p-3 border-b-2 border-gray">Name</th>
                <th className="text-left text-gray p-3 border-b-2 border-gray">Link</th>
              </tr>
            </thead>
            <tbody>
              {group.links.map((link, linkIndex) => (
                <tr
                  key={linkIndex}
                  className={linkIndex % 2 === 0 ? 'bg-lightGray' : 'bg-white'}
                >
                  <td className="p-3 border border-gray text-navy">{link.name}</td>
                  <td className="p-3 border border-gray">
                    <a
                      href={link.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-teal underline hover:text-navy"
                    >
                      {link.url}
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
};

export default LinksPage;
