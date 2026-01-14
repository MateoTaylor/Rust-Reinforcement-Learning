
using System;
using System.Collections.Generic;
using System.Linq;
using Oxide.Core;
using Oxide.Core.Plugins;
using UnityEngine;

namespace Oxide.Plugins
{
    [Info("GiveItem", "You", "0.1")]
    [Description("Helper functions for giving player a pickaxe")]
    public class GiveItem : CovalencePlugin
    {
        public string GivePickaxe()
        {
            try
            {
                Item pickaxe = ItemManager.CreateByName("pickaxe", 1);
                if (pickaxe == null)
                {
                    return "ERR|create_pickaxe_failed";
                }
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
                if (!player.inventory.GiveItem(pickaxe))
                {
                    pickaxe.Remove();
                    return "ERR|give_pickaxe_failed";
                }
                return $"OK|pickaxe_given|player:{player.displayName}";
            }
            catch (Exception ex)
            {
                PrintError($"GiveItem: GiveItem failed: {ex.Message}");
                return $"ERR|give_item_failed:{ex.Message}";
            }
        }


    }
}