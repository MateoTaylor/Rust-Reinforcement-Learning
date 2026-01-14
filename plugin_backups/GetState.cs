// File: GetState.cs
// Drop into: <your_server>/oxide/plugins/
// Helper functions for building game state snapshots (players and ore nodes)

using System;
using System.Collections.Generic;
using System.Linq;
using Oxide.Core;
using Oxide.Core.Plugins;
using UnityEngine;

namespace Oxide.Plugins
{
    [Info("GetState", "You", "0.1")]
    [Description("Helper functions for building game state snapshots")]
    public class GetState : CovalencePlugin
    {
        // Limit nodes to avoid enormous payloads on dense maps
        private const int MAX_NODES_RETURNED = 1000;

        /// <summary>
        /// Build the structured snapshot containing players and ore nodes.
        /// This method MUST be called on the main thread.
        /// </summary>
        /// <returns>Snapshot object containing players and nodes data</returns>
        public object BuildSnapshot()
        {
            var playersList = new List<object>();
            try
            {
                // Copy to list to avoid collection-modified exceptions
                var active = BasePlayer.activePlayerList?.ToList() ?? new List<BasePlayer>();
                foreach (var p in active)
                {
                    if (p == null) continue;
                    try
                    {
                        // Basic spatial & identity
                        var pos = p.transform?.position ?? Vector3.zero;
                        var rot = p.transform?.eulerAngles ?? Vector3.zero;

                        // Health & metabolism (defensive)
                        float? health = null;
                        try { health = p.Health(); } catch { }

                        float? maxHealth = null;
                        try
                        {
                            // MaxHealth is not guaranteed across versions; call defensively
                            var mi = p.GetType().GetMethod("MaxHealth");
                            if (mi != null) maxHealth = Convert.ToSingle(mi.Invoke(p, null));
                        }
                        catch { }

                        float? calories = null;
                        float? hydration = null;
                        bool isSwimming = false;
                        try
                        {
                            var meta = p.metabolism;
                            if (meta != null)
                            {
                                try { calories = (float?)meta.calories?.value; } catch { }
                                try { hydration = (float?)meta.hydration?.value; } catch { }
                            }
                        }
                        catch { }
                        
                        // Check if player is swimming using WaterFactor
                        try
                        {
                            float waterFactor = p.WaterFactor();
                            // Player is swimming if water factor is above 0.65 (standard swimming threshold)
                            isSwimming = waterFactor > 0.30f;
                        }
                        catch { }

                        // Inventory aggregation (main, belt, wear)
                        var items = new List<object>();
                        try
                        {
                            var main = p.inventory?.containerMain?.itemList;
                            if (main != null)
                            {
                                foreach (var it in main)
                                {
                                    var serial = SerializeItem(it, "main");
                                    if (serial != null) items.Add(serial);
                                }
                            }
                        }
                        catch { /* ignore inventory exceptions per-player */ }

                        try
                        {
                            var belt = p.inventory?.containerBelt?.itemList;
                            if (belt != null)
                            {
                                foreach (var it in belt)
                                {
                                    var serial = SerializeItem(it, "belt");
                                    if (serial != null) items.Add(serial);
                                }
                            }
                        }
                        catch { }

                        try
                        {
                            var wear = p.inventory?.containerWear?.itemList;
                            if (wear != null)
                            {
                                foreach (var it in wear)
                                {
                                    var serial = SerializeItem(it, "wear");
                                    if (serial != null) items.Add(serial);
                                }
                            }
                        }
                        catch { }

                        playersList.Add(new
                        {
                            steamId = p.userID.ToString(),
                            name = p.displayName,
                            position = new { x = pos.x, y = pos.y, z = pos.z },
                            rotation = new { x = rot.x, y = rot.y, z = rot.z },
                            health = health,
                            maxHealth = maxHealth,
                            calories = calories,
                            hydration = hydration,
                            isSleeping = p.IsSleeping(),
                            isConnected = p.IsConnected,
                            isSwimming = isSwimming,
                            items = items
                        });
                    }
                    catch (Exception ex)
                    {
                        PrintWarning($"GetState: error building player snapshot for a player: {ex}");
                        // continue with other players
                    }
                }
            }
            catch (Exception ex)
            {
                PrintWarning($"GetState: players collection error: {ex}");
            }

            // Nodes: search server entities and pick OreResourceEntity instances (stone/metal/sulfur)
            var nodesList = new List<object>();
            var nodePositions = new List<Vector3>(); // Store node positions for NodeInView calculation
            var playerObjects = new Dictionary<string, BasePlayer>(); // Store player references for NodeInView
            
            // Extract player objects for later use
            foreach (var p in BasePlayer.activePlayerList?.ToList() ?? new List<BasePlayer>())
            {
                if (p != null)
                {
                    playerObjects[p.userID.ToString()] = p;
                }
            }
            
            try
            {
                var entities = BaseNetworkable.serverEntities?.ToList() ?? new List<BaseNetworkable>();
                int count = 0;
                foreach (var net in entities)
                {
                    if (count >= MAX_NODES_RETURNED) break;
                    try
                    {
                        var be = net as BaseEntity;
                        if (be == null) continue;

                        // Check for the OreResourceEntity component (defensive)
                        var ore = be.GetComponent<global::OreResourceEntity>(); // fully qualified to avoid ambiguity
                        if (ore == null) continue;

                        // Determine prefab name (ShortPrefabName is common)
                        string prefab = null;
                        try { prefab = be.ShortPrefabName; } catch { }
                        if (string.IsNullOrEmpty(prefab))
                        {
                            try { prefab = be.name; } catch { }
                        }
                        if (string.IsNullOrEmpty(prefab)) prefab = "unknown";

                        // Filter: only include typical ore node prefabs (stone, metal, sulfur)
                        var lower = prefab.ToLowerInvariant();
                        if (!(lower.Contains("stone") || lower.Contains("metal") || lower.Contains("sulfur") || lower.Contains("ore"))) continue;

                        var pos = be.transform?.position ?? Vector3.zero;
                        float? nodeHealth = TryGetEntityHealth(be); // defensive health read

                        nodesList.Add(new
                        {
                            prefab = prefab,
                            position = new { x = pos.x, y = pos.y, z = pos.z },
                            health = nodeHealth
                        });
                        
                        nodePositions.Add(pos); // Store for NodeInView calculation

                        count++;
                    }
                    catch
                    {
                        // skip problematic entities
                    }
                }
            }
            catch (Exception ex)
            {
                PrintWarning($"GetState: nodes collection error: {ex}");
            }

            // Calculate NodeInView for each player and rebuild player list
            var finalPlayersList = new List<object>();
            foreach (var playerData in playersList)
            {
                int nodeInView = 0;
                string steamId = null;
                string name = null;
                object position = null;
                object rotation = null;
                float? health = null;
                float? maxHealth = null;
                float? calories = null;
                float? hydration = null;
                bool isSleeping = false;
                bool isConnected = false;
                bool isSwimming = false;
                List<object> items = null;
                
                try
                {
                    // Extract data using reflection to avoid dynamic
                    var type = playerData.GetType();
                    steamId = type.GetProperty("steamId")?.GetValue(playerData) as string;
                    name = type.GetProperty("name")?.GetValue(playerData) as string;
                    position = type.GetProperty("position")?.GetValue(playerData);
                    rotation = type.GetProperty("rotation")?.GetValue(playerData);
                    health = type.GetProperty("health")?.GetValue(playerData) as float?;
                    maxHealth = type.GetProperty("maxHealth")?.GetValue(playerData) as float?;
                    calories = type.GetProperty("calories")?.GetValue(playerData) as float?;
                    hydration = type.GetProperty("hydration")?.GetValue(playerData) as float?;
                    
                    var isSleepingObj = type.GetProperty("isSleeping")?.GetValue(playerData);
                    if (isSleepingObj != null) isSleeping = (bool)isSleepingObj;
                    
                    var isConnectedObj = type.GetProperty("isConnected")?.GetValue(playerData);
                    if (isConnectedObj != null) isConnected = (bool)isConnectedObj;
                    
                    var isSwimmingObj = type.GetProperty("isSwimming")?.GetValue(playerData);
                    if (isSwimmingObj != null) isSwimming = (bool)isSwimmingObj;

                    items = type.GetProperty("items")?.GetValue(playerData) as List<object>;
                    
                    // Calculate NodeInView
                    if (!string.IsNullOrEmpty(steamId) && playerObjects.ContainsKey(steamId))
                    {
                        var player = playerObjects[steamId];
                        if (player != null && !player.IsSleeping())
                        {
                            nodeInView = CalculateNodeInView(player, nodePositions);
                        }
                    }
                }
                catch (Exception ex)
                {
                    PrintWarning($"GetState: error calculating NodeInView: {ex}");
                }
                
                finalPlayersList.Add(new
                {
                    steamId = steamId,
                    name = name,
                    position = position,
                    rotation = rotation,
                    health = health,
                    maxHealth = maxHealth,
                    calories = calories,
                    hydration = hydration,
                    isSleeping = isSleeping,
                    isConnected = isConnected,
                    items = items,
                    nodeInView = nodeInView,
                    isSwimming = isSwimming
                });
            }

            var snapshot = new
            {
                serverTimeUtc = DateTime.UtcNow,
                players = finalPlayersList,
                nodes = nodesList,
                playersCount = finalPlayersList.Count,
                nodesCount = nodesList.Count
            };

            return snapshot;
        }

        /// <summary>
        /// Serialize an Item to a compact object; runs on main thread.
        /// </summary>
        private object SerializeItem(Item item, string container)
        {
            if (item == null) return null;
            try
            {
                string shortname = null;
                try { shortname = item.info?.shortname; } catch { }
                string displayName = null;
                try { displayName = item.info?.displayName?.english; } catch { }
                int amount = 0;
                try { amount = item.amount; } catch { }
                ulong skin = 0;
                try { skin = item.skin; } catch { }
                float? condition = null;
                try { condition = item.condition; } catch { }

                int? itemId = null;
                try { itemId = item.info?.itemid; } catch { }

                return new
                {
                    container = container,
                    itemId = itemId,
                    shortName = shortname,
                    displayName = displayName,
                    amount = amount,
                    skin = skin,
                    condition = condition
                };
            }
            catch
            {
                return new { container = container, shortName = "unknown" };
            }
        }

        /// <summary>
        /// Calculate if any node is visible to the player (within view cone and line of sight).
        /// Returns 1 if at least one node is visible, 0 otherwise.
        /// </summary>
        private int CalculateNodeInView(BasePlayer player, List<Vector3> nodePositions)
        {
            if (player == null || nodePositions == null || nodePositions.Count == 0)
                return 0;
            
            try
            {
                // Get player's eye position (head) and view direction
                Vector3 eyePos = player.transform.position;
                
                // Field of view parameters
                float fovAngle = 0.7f; // Typical FoV in degrees (half angle, so 90 total cone)
                float maxViewDistance = 30f; // Maximum distance to check for nodes

                int layer_mask = LayerMask.GetMask("Construction", "Default", "Deployed", "Resource", "Terrain", "World");                
                foreach (var nodePos in nodePositions)
                {
                    // Calculate direction from player to node
                    Vector3 toNode = nodePos - eyePos;
                    float distance = toNode.magnitude;
                        
                    // Skip if too far
                    if (distance > maxViewDistance)
                        continue;
                        
                        // Check if node is within view cone                    
                    Vector3 directionToNode = (nodePos - player.eyes.position).normalized;
                    float dot = Vector3.Dot(player.eyes.HeadForward(), directionToNode);

                    if (dot >= fovAngle)
                    {
                            // Node is in view cone, now check line of sight
                        RaycastHit hit;
                        if (Physics.Raycast(player.eyes.position, directionToNode, out hit, distance, layer_mask))
                        {
                            var entity = hit.GetEntity();
                            if (entity != null && Vector3.Distance(hit.point, nodePos) < 3f) {
                                // Node is actually visible and not blocked
                                return 1;
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                PrintWarning($"GetState: CalculateNodeInView error: {ex}");
                return 2;
            }
            
            return 0; // No visible nodes found
        }
        
        /// <summary>
        /// Try to read an entity's health in a defensive manner.
        /// </summary>
        private float? TryGetEntityHealth(BaseEntity be)
        {
            if (be == null) return null;
            try
            {
                // Most BaseEntity-derived types expose a health field or property:
                try
                {
                    // direct property access (most common)
                    var h = be.Health();
                    return h;
                }
                catch { }

                // Fallback: reflect for "health" or "GetHealth"/"Health" methods/properties
                try
                {
                    var t = be.GetType();
                    var prop = t.GetProperty("health");
                    if (prop != null)
                    {
                        var val = prop.GetValue(be);
                        if (val is float fv) return fv;
                        if (val is double dv) return (float)dv;
                        if (val is int iv) return iv;
                    }
                }
                catch { }

                try
                {
                    var mi = be.GetType().GetMethod("GetHealth") ?? be.GetType().GetMethod("Health") ?? be.GetType().GetMethod("health");
                    if (mi != null)
                    {
                        var val = mi.Invoke(be, null);
                        if (val is float vf) return vf;
                        if (val is double vd) return (float)vd;
                        if (val is int vi) return vi;
                    }
                }
                catch { }
            }
            catch { }
            return null;
        }
    }
}
