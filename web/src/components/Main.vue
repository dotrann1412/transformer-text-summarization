<template>
  <div class="container">
    <el-input
      class="text-container"
      v-model="textarea1"
      type="textarea"
      rows="40"
      placeholder="Please input"
    />

    <el-button type="primary" round :disabled="textarea1.length == 0">Summarize</el-button>

    <el-input
      class="text-container"
      v-model="textarea2"
      type="textarea"
      rows="40"
      placeholder="Summarized text"
      disabled
    />
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
const textarea1 = ref('')
const textarea2 = ref('')
</script>

<style scoped lang="scss">
.container {
  margin: auto;
  position: relative;
  display: flex;
  flex-direction: row;
  align-items: center;
  .text-container {
    padding: 30px;
    width: 45vw;
  }
}
</style>
