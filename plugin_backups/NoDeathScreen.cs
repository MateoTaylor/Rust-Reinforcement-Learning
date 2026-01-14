/*
 ########### README ####################################################
                                                                             
  !!! DON'T EDIT THIS FILE !!!

    Discord: paulsimik#0506
                                                                     
 ########### CHANGES ###################################################

 1.0.2
    - Rewrited the plugin
    - Added configuration
    - Added option player has sleeping bag
    - Added option end sleeping after player respawned
    - Added command for toggle auto respawn
    - Added auto respawn after player connected

 #######################################################################
*/

using System.Collections.Generic;
using Oxide.Core;
using System.Linq;
using System;
using Newtonsoft.Json;

namespace Oxide.Plugins
{
    [Info("No Death Screen", "Paulsimik", "1.0.2")]
    [Description("Disables the death screen by automatically respawning players")]
    public class NoDeathScreen : RustPlugin
    {
        #region [Fields]

        private const string permUse = "nodeathscreen.use";
        private const string permToggle = "nodeathscreen.toggle";
        private const string fileName = "NoDeathScreen";

        private Configuration config;
        private List<string> playerData = new List<string>();

        #endregion

        #region [Oxide Hooks]

        private void Init()
        {
            permission.RegisterPermission(permUse, this);
            permission.RegisterPermission(permToggle, this);
            LoadData();

            foreach (var command in config.chatCommands)
            {
                cmd.AddChatCommand(command, this, nameof(chatCmdNoDeathScreen));
            }
        }

        private void Unload() => SaveData();

        private void OnPlayerConnected(BasePlayer player)
        {
            if (player.HasPlayerFlag(BasePlayer.PlayerFlags.ReceivingSnapshot))
            {
                timer.Once(1f, () => OnPlayerConnected(player));
                return;
            }

            ProcessRespawn(player);
        }

        private void OnEntityDeath(BasePlayer player)
        {
            if (player.IsNpc)
                return;

            ProcessRespawn(player);
        }

        #endregion

        #region [Hooks]   

        private void ProcessRespawn(BasePlayer player)
        {
            if (!HasPermRespawn(player.UserIDString))
                return;

            if (playerData.Contains(player.UserIDString))
                return;

            NextTick(() =>
            {
                if (player == null || !player.IsConnected)
                    return;

                if (player.IsDead())
                    player.Respawn();

                if (config.endSleeping && player.IsSleeping())
                    player.EndSleeping();
            });
        }

        private bool HasPermRespawn(string playerID)
        {
            return permission.UserHasPermission(playerID, permUse);
        }

        #endregion

        #region [Chat Commands]

        private void chatCmdNoDeathScreen(BasePlayer player, string command, string[] args)
        {
            if (!permission.UserHasPermission(player.UserIDString, permToggle))
            {
                SendReply(player, GetLang("noPerm", player.UserIDString));
                return;
            }

            if (playerData.Contains(player.UserIDString))
            {
                playerData.Remove(player.UserIDString);
                SendReply(player, GetLang("enabled", player.UserIDString));
            }
            else
            {
                playerData.Add(player.UserIDString);
                SendReply(player, GetLang("disabled", player.UserIDString));
            }
        }

        #endregion

        #region [Classes]

        private class Configuration
        {
            [JsonProperty(PropertyName = "No respawn if the player has a sleeping bag or bed")]
            public bool sleepingBagBypass;

            [JsonProperty(PropertyName = "End sleeping after respawned")]
            public bool endSleeping;

            [JsonProperty(PropertyName = "Custom chat commands")]
            public string[] chatCommands;

            public VersionNumber version;
        }

        #endregion

        #region [Data]

        private void SaveData() => Interface.Oxide.DataFileSystem.WriteObject(fileName, playerData);

        private void LoadData()
        {
            try
            {
                playerData = Interface.Oxide.DataFileSystem.ReadObject<List<string>>(fileName);
            }
            catch (Exception e)
            {
                PrintError($"Data was not loaded!");
                PrintError(e.Message);
            }

            SaveData();
        }

        #endregion

        #region [Config]

        private Configuration GetDefaultConfig()
        {
            return new Configuration
            {
                sleepingBagBypass = true,
                endSleeping = true,
                chatCommands = new string[]
                {
                    "nodeathscreen",
                    "nds",
                    "ns",
                    "autorespawn",
                    "atr",
                },
                version = Version
            };
        }

        protected override void LoadDefaultConfig()
        {
            config = GetDefaultConfig();
            Puts("Generating new configuration file........");
        }

        protected override void SaveConfig() => Config.WriteObject(config);

        protected override void LoadConfig()
        {
            base.LoadConfig();

            try
            {
                config = Config.ReadObject<Configuration>();

                if (config == null)
                {
                    LoadDefaultConfig();
                }
            }
            catch
            {
                PrintError("######### Configuration file is not valid! #########");
                return;
            }

            SaveConfig();
        }

        #endregion

        #region [Localization]

        private string GetLang(string key, string playerID) => string.Format(lang.GetMessage(key, this, playerID));

        protected override void LoadDefaultMessages()
        {
            lang.RegisterMessages(new Dictionary<string, string>
            {
                { "noPerm", "You don't have permissions" },
                { "enabled", "No Death Screen has been enabled" },
                { "disabled", "No Death Screen has been disabled" }

            }, this);
        }

        #endregion
    }
}