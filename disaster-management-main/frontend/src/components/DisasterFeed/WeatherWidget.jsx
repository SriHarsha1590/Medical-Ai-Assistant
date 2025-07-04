import React from 'react';

const weatherBackgrounds = {
  Clear: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Clouds: "https://images.unsplash.com/photo-1502082553048-f009c37129b9?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Rain: "https://images.unsplash.com/photo-1501612780327-4504553879a2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Snow: "https://images.unsplash.com/photo-1517313887206-7fbb419c9a9e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Thunderstorm: "https://images.unsplash.com/photo-1508818319268-d31f45d14e41?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Drizzle: "https://images.unsplash.com/photo-1561488110-2e41f61e2826?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Mist: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Smoke: "https://images.unsplash.com/photo-1509715695397-83d4da1b85fa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Haze: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Dust: "https://images.unsplash.com/photo-1502082553048-f009c37129b9?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Fog: "https://images.unsplash.com/photo-1517511620798-cec17d428bc0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Sand: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Ash: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Squall: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
  Tornado: "https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
};

const WeatherWidget = ({ weather }) => {
  const getBackgroundImage = (weatherCondition) => {
    const backgroundImage =
      weatherBackgrounds[weatherCondition] ||
      "https://images.unsplash.com/photo-1499346030926-9a72daac6c63?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80";
    return `linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url(${backgroundImage})`;
  };

  return (
    <div className="weather-widget flex flex-col items-center p-4 rounded-lg shadow-lg" style={{
      backgroundImage: getBackgroundImage(weather?.weather[0].main),
      backgroundSize: "cover",
      backgroundPosition: "center",
      width: "100%",
      height: "75%",
      color: "white",
      minWidth: "200px",
    }}>
      {weather && (
        <>
          {/* City Name */}
          <h2 className="text-2xl font-semibold">{weather.name}</h2>
          {/* Weather Icon */}
          <img
            src={`https://openweathermap.org/img/wn/${weather.weather[0].icon}@2x.png`}
            alt={weather.weather[0].description}
            className="w-20 h-20"
          />
          {/* Description */}
          <p className="capitalize text-sm mb-2">{weather.weather[0].description}</p>
          {/* Temperature */}
          <p className="text-3xl font-bold mb-2">{weather.main.temp}°C</p>
          {/* Additional Info */}
          <div className="flex flex-col items-center space-y-1">
            <p className="text-sm">Humidity: {weather.main.humidity}%</p>
            <p className="text-sm">Wind Speed: {weather.wind.speed} m/s</p>
          </div>
        </>
      )}
    </div>
  );
};

export default WeatherWidget;
