// File: TrainingEnvHelper.cs
// Drop into: <your_server>/oxide/plugins/
// Helper functions for managing YOLO training environment

using System;
using System.Collections.Generic;
using System.Linq;
using Oxide.Core;
using Oxide.Core.Plugins;
using UnityEngine;

namespace Oxide.Plugins
{
    [Info("ResetEnv", "You", "0.1")]
    [Description("Helper functions for managing YOLO training environment")]
    public class ResetEnv : CovalencePlugin
    {
        /// <summary>
        /// Resets all resource nodes on the server by killing existing nodes and spawning new ones.
        /// </summary>
        /// <param name="nodesToSpawn">Number of new nodes to spawn (default: 300)</param>
        /// <returns>Status message with counts</returns>
        private string ResetNodes(int nodesToSpawn = 300)
        {
            try
            {
                int killedCount = 0;
                int spawnedCount = 0;
                
                // Get all server entities
                var entities = BaseNetworkable.serverEntities?.ToList() ?? new List<BaseNetworkable>();
                
                // Kill all existing ore resource nodes
                foreach (var net in entities)
                {
                    try
                    {
                        if (net is not BaseEntity be) continue;
                        
                        // Check if it's an ore resource entity
                        var ore = be.GetComponent<global::OreResourceEntity>();
                        if (ore == null) continue;
                        
                        // Kill the node
                        be.Kill();
                        killedCount++;
                    }
                    catch (Exception ex)
                    {
                        PrintWarning($"ResetEnv: Error killing node: {ex.Message}");
                    }
                }
                
                Puts($"ResetEnv: Killed {killedCount} existing nodes");
                
                // Spawn new nodes
                // Note: We need to get valid spawn positions from the map
                // For now, we'll spawn nodes in a grid pattern around the map center
                var mapSize = TerrainMeta.Size.x;
                var center = TerrainMeta.Center;
                var random = new System.Random();
                
                // Node prefab paths (common ore node types)
                var nodePrefabs = new List<string>
                {
                    "assets/bundled/prefabs/autospawn/resource/ores/stone-ore.prefab",
                    "assets/bundled/prefabs/autospawn/resource/ores/metal-ore.prefab",
                    "assets/bundled/prefabs/autospawn/resource/ores/sulfur-ore.prefab"
                };
                
                int attempts = 0;
                int maxAttempts = nodesToSpawn * 10; // Allow multiple attempts to find valid positions
                
                while (spawnedCount < nodesToSpawn && attempts < maxAttempts)
                {
                    attempts++;
                    try
                    {
                        // Random position within 1/4 of the map size from center
                        float randomX = 0.25f * center.x + (float)(random.NextDouble() * mapSize - mapSize / 2);
                        float randomZ = 0.25f * center.z + (float)(random.NextDouble() * mapSize - mapSize / 2);
                        
                        // Get terrain height at this position
                        float terrainHeight = TerrainMeta.HeightMap.GetHeight(new Vector3(randomX, 0, randomZ));
                        Vector3 spawnPos = new Vector3(randomX, terrainHeight, randomZ);
                        
                        // Check if position is above water level
                        if (terrainHeight < TerrainMeta.WaterMap.GetHeight(spawnPos))
                        {
                            continue; // Skip underwater positions
                        }
                        
                        // Check if topology at this position is Field
                        int topology = TerrainMeta.TopologyMap.GetTopology(spawnPos);
                        if ((topology & (int)TerrainTopology.Enum.Field) == 0)
                        {
                            continue; // Skip non-field positions
                        }
                                                
                        // Randomly select a node type
                        string prefab = nodePrefabs[random.Next(nodePrefabs.Count)];
                        
                        // Spawn the entity
                        var entity = GameManager.server.CreateEntity(prefab, spawnPos, Quaternion.identity);
                        if (entity != null)
                        {
                            entity.Spawn();
                            spawnedCount++;
                        }
                    }
                    catch (Exception ex)
                    {
                        PrintWarning($"ResetEnv: Error spawning node (attempt {attempts}): {ex.Message}");
                    }
                }
                Puts($"ResetEnv: Spawned {spawnedCount} new nodes");
                
                return $"OK|nodes_reset|killed:{killedCount}|spawned:{spawnedCount}";
            }
            catch (Exception ex)
            {
                PrintError($"ResetEnv: ResetNodes failed: {ex.Message}");
                return $"ERR|reset_nodes_failed:{ex.Message}";
            }
        }
        
        /// <summary>
        /// Clears all inventory items from the first active player found on the server.
        /// Assumes only one player is on the server at a time.
        /// </summary>
        /// <returns>Status message indicating success or failure</returns>
        private string ClearInventory()
        {
            try
            {
                int totalItemsCleared = 0;
                
                // Get the first active player (assuming single player on server)
                var active = BasePlayer.activePlayerList?.ToList() ?? new List<BasePlayer>();
                if (active.Count == 0)
                {
                    return "ERR|no_active_players";
                }
                
                var player = active[0];
                if (player == null)
                {
                    return "ERR|player_null";
                }
                
                Puts($"ResetEnv: Clearing inventory for player {player.displayName} ({player.userID})");
                
                // Clear main inventory
                try
                {
                    var main = player.inventory?.containerMain;
                    if (main != null && main.itemList != null)
                    {
                        int count = main.itemList.Count;
                        main.Clear();
                        totalItemsCleared += count;
                        Puts($"ResetEnv: Cleared {count} items from main inventory");
                    }
                }
                catch (Exception ex)
                {
                    PrintWarning($"ResetEnv: Error clearing main inventory: {ex.Message}");
                }
                
                // Clear belt inventory
                try
                {
                    var belt = player.inventory?.containerBelt;
                    if (belt != null && belt.itemList != null)
                    {
                        int count = belt.itemList.Count;
                        belt.Clear();
                        totalItemsCleared += count;
                        Puts($"ResetEnv: Cleared {count} items from belt");
                    }
                }
                catch (Exception ex)
                {
                    PrintWarning($"ResetEnv: Error clearing belt: {ex.Message}");
                }
                
                // Clear wear inventory (clothing/armor)
                try
                {
                    var wear = player.inventory?.containerWear;
                    if (wear != null && wear.itemList != null)
                    {
                        int count = wear.itemList.Count;
                        wear.Clear();
                        totalItemsCleared += count;
                        Puts($"ResetEnv: Cleared {count} items from wear");
                    }
                }
                catch (Exception ex)
                {
                    PrintWarning($"ResetEnv: Error clearing wear: {ex.Message}");
                }
                
                Puts($"ResetEnv: Total items cleared: {totalItemsCleared}");
                
                return $"OK|inventory_cleared|items:{totalItemsCleared}|player:{player.displayName}";
            }
            catch (Exception ex)
            {
                PrintError($"ResetEnv: ClearInventory failed: {ex.Message}");
                return $"ERR|clear_inventory_failed:{ex.Message}";
            }
        }

        /// <summary>
        /// Kills the first active player found on the server.
        /// Assumes only one player is on the server at a time.
        /// </summary>
        /// <returns>Status message indicating success or failure</returns>
        private string KillPlayer()
        {
            try
            {
                // Get the first active player (assuming single player on server)
                var active = BasePlayer.activePlayerList?.ToList() ?? new List<BasePlayer>();
                if (active.Count == 0)
                {
                    return "ERR|no_active_players";
                }

                var player = active[0];
                if (player == null)
                {
                    return "ERR|player_null";
                }

                Puts($"ResetEnv: Killing player {player.displayName} ({player.userID})");

                // Check if player is already dead
                if (player.IsDead())
                {
                    Puts($"ResetEnv: Player {player.displayName} is already dead");
                    return $"OK|player_already_dead|player:{player.displayName}";
                }

                // Kill the player using Die() method
                try
                {
                    player.Die();
                    Puts($"ResetEnv: Successfully killed player {player.displayName}");
                    return $"OK|player_killed|player:{player.displayName}";
                }
                catch (Exception ex)
                {
                    PrintWarning($"ResetEnv: Error calling Die() on player: {ex.Message}");

                    // Fallback: set health to 0
                    try
                    {
                        player.Hurt(player.Health() + 100f);
                        Puts($"ResetEnv: Killed player {player.displayName} using Hurt() method");
                        return $"OK|player_killed|player:{player.displayName}|method:hurt";
                    }
                    catch (Exception ex2)
                    {
                        PrintError($"ResetEnv: Fallback Hurt() method also failed: {ex2.Message}");
                        return $"ERR|kill_failed:{ex2.Message}";
                    }
                }
            }
            catch (Exception ex)
            {
                PrintError($"ResetEnv: KillPlayer failed: {ex.Message}");
                return $"ERR|kill_player_failed:{ex.Message}";
            }
        }
        
    

        /// <summary>
        /// Resets the training environment by clearing train/test splits and moving all data back to 'all' folders.
        /// This allows for a fresh train/test split to be performed.
        /// </summary>
        /// <returns>Status message indicating success or failure</returns>
        public string ResetTrainingEnvironment()
        {
            try
            {
                Puts("ResetEnv: ResetTrainingEnvironment called");
                
                // Step 1: Reset all resource nodes on the server
                string nodeResetResult = ResetNodes(200);
                Puts($"ResetEnv: Node reset result: {nodeResetResult}");
                
                // Step 2: Clear player inventory
                string inventoryClearResult = ClearInventory();
                Puts($"ResetEnv: Inventory clear result: {inventoryClearResult}");
                
                // Step 3: Kill the player
                string killPlayerResult = KillPlayer();
                Puts($"ResetEnv: Kill player result: {killPlayerResult}");

                return $"OK|training_env_reset|{nodeResetResult}|{inventoryClearResult}|{killPlayerResult}";
            }
            catch (Exception ex)
            {
                PrintError($"ResetEnv: ResetTrainingEnvironment failed: {ex.Message}");
                return $"ERR|reset_failed:{ex.Message}";
            }
        }
    }
}
