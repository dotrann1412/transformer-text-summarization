import axios from 'axios'

const service = axios.create({
    baseURL: import.meta.env.VITE_URL,
    timeout: 60000 * 5
})

service.interceptors.request.use(
    async (config) => {
        return config
    },
    (error) => {
        Promise.reject(error)
    }
)

service.interceptors.response.use(
    (response) => {
        return response
    },
    async (error) => {
        console.log(error)
    }
)

export default service