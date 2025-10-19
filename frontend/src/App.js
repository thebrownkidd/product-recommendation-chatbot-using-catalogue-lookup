import React, { useState } from 'react';
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';
import Header from './components/Header';
import TextSearch from './components/TextSearch';
import ImageSearch from './components/ImageSearch';

function App() {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="app">
      <Header />
      <div className="container">
        <Tabs selectedIndex={activeTab} onSelect={index => setActiveTab(index)}>
          <TabList>
            <Tab>Text Search</Tab>
            <Tab>Image Search</Tab>
          </TabList>

          <TabPanel>
            <TextSearch />
          </TabPanel>
          <TabPanel>
            <ImageSearch />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  );
}

export default App;