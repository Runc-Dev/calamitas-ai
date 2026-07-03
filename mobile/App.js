import React from "react";
import { StatusBar } from "expo-status-bar";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import HomeScreen from "./src/screens/HomeScreen";
import ResultScreen from "./src/screens/ResultScreen";
import { APP_RED } from "./src/utils/colors";

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Stack.Navigator
        screenOptions={{
          headerStyle: { backgroundColor: APP_RED },
          headerTintColor: "#fff",
          headerTitleStyle: { fontWeight: "800" },
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: "AFETSONAR" }}
        />
        <Stack.Screen
          name="Result"
          component={ResultScreen}
          options={{ title: "Damage Assessment" }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
