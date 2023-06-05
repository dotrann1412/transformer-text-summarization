<template>
  <el-config-provider namespace="ep">
    <Logos />
    <div class="flex main-container">
      <Main />
    </div>
  </el-config-provider>
</template>

<style>
#app {
  text-align: center;
  color: var(--ep-text-color-primary);
}

.main-container {
  margin-top: 30px;
  height: calc(100vh - var(--ep-menu-item-height) - 3px);
}
</style>
