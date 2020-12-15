/*****************************************************************//**
 * \file   MappedBuffer.h
 * \brief  A wrapper for OpenCL mapped buffer
 * 
 * \author Wenhao Li
 * \date   September 2020
 *********************************************************************/
#pragma once

#include <CL/opencl.hpp>
#include "Error.hpp"


enum class AccessMode
{
    Read,
    Write,
    ReadWrite,
    NotSpecified
};

constexpr cl_map_flags GetCLMapFlag(AccessMode mode)
{
    switch(mode)
    {
    case AccessMode::Read:
        return CL_MAP_READ;
    case AccessMode::Write:
        return CL_MAP_WRITE;
    case AccessMode::ReadWrite:
        return CL_MAP_READ | CL_MAP_WRITE;
    }
    throw InvalidAccessMode{};
}

constexpr cl_mem_flags GetCLMemFlag(AccessMode mode)
{
    switch(mode)
    {
    case AccessMode::Read:
        return CL_MEM_READ_ONLY;
    case AccessMode::Write:
        return CL_MEM_WRITE_ONLY;
    case AccessMode::ReadWrite:
        return CL_MEM_READ_WRITE;   
    }
    throw InvalidAccessMode{};
}

template<typename T, AccessMode mode>
class MappedBuffer
{
    cl::CommandQueue& m_queue;
    cl::Buffer const& m_buffer_ref;
public:
    T* m_ptr;
    MappedBuffer(cl::CommandQueue& queue, T* ptr, cl::Buffer const& buffer_ref):m_queue(queue), m_ptr(ptr), m_buffer_ref(buffer_ref){}

    //template<typename = std::enable_if_t<mode==AccessMode::Write||mode==AccessMode::ReadWrite>>
    operator T* () noexcept { return m_ptr; }

    operator const T* () const noexcept { return m_ptr; }

    //template<typename = std::enable_if_t<mode == AccessMode::Write || mode == AccessMode::ReadWrite>>
    auto get()
    {
        return m_ptr;
    }

    auto get() const
    {
        return m_ptr;
    }

    auto& operator[](size_t index)
    {
        return m_ptr[index];
    }

    ~MappedBuffer()
    {
        m_queue.enqueueUnmapMemObject(m_buffer_ref, m_ptr);
    }

    /*deleted special member function */
    MappedBuffer(MappedBuffer const&) = delete;
    MappedBuffer(MappedBuffer&&) = delete;
    MappedBuffer& operator=(MappedBuffer const&) = delete;
    MappedBuffer& operator=(MappedBuffer&&) = delete;
};


template<typename T>
class Buffer:cl::Buffer
{
    cl::CommandQueue& m_queue;
    AccessMode m_mode;
public:
    auto& getClBuffer()
    {
        return static_cast<cl::Buffer&>(*this);
    }
    auto& getClBuffer() const
    {
        return static_cast<cl::Buffer const&>(*this);
    }
    auto getSize() const
    {
        return getClBuffer().template getInfo<CL_MEM_SIZE>();
    }

    using value_type = T;

    /**
     * Create an empty buffer without allocation
     * 
     * \param queue
     * \param mode
     */
    Buffer(cl::CommandQueue& queue, AccessMode mode = AccessMode::NotSpecified):m_queue(queue), m_mode(mode){}

    /**
     * @brief Create an empty buffer on the device
     */
    Buffer(size_t size, cl::Context& context, cl::CommandQueue& queue, AccessMode mode)
        : cl::Buffer{ context, GetCLMemFlag(mode), sizeof(T) * size, nullptr },
        m_queue(queue),
        m_mode(mode)
    {}

    /**
     * @brief Create a buffer on the device and copy the data from host
     */
    Buffer(size_t size, cl::Context& context, T const* const data, cl::CommandQueue& queue, AccessMode mode)
        : cl::Buffer{ context, GetCLMemFlag(mode) | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, const_cast<T*>(data) },
        m_queue(queue),
        m_mode(mode)
    {}


    /**
     * @brief Create an empty buffer on the device, with extra flags
     * @param extraFlags can be CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR
     */
    Buffer(size_t size, cl::Context& context, cl::CommandQueue& queue, AccessMode mode, int extraFlags, T* const data = nullptr)
        :cl::Buffer{context, GetCLMemFlag(mode) | extraFlags, sizeof(T)*size, data},
        m_queue(queue),
        m_mode(mode)
    {}


    //~Buffer()
    //    //= default;
    //{
    //    if (m_mode != AccessMode::NotSpecified)
    //    {
    //        cl::Buffer::release();
    //    }
    //}


    template<AccessMode mode>
    auto map(bool blocking = true)
    {
        return MappedBuffer<T, mode>
        {
            m_queue,
            static_cast<T*>(m_queue.enqueueMapBuffer(getClBuffer(), blocking, GetCLMapFlag(mode), 0, getSize())),
            getClBuffer()
        };
    }

    /*special member functions*/
    /**
     * @brief Copy a buffer to a specified context and specified command queue
     */
    Buffer(Buffer const& rhs, cl::Context& context, cl::CommandQueue& queue)
        :cl::Buffer{context, GetCLMemFlag(rhs.m_mode), rhs.getSize(), nullptr},
        m_queue(queue),
        m_mode(rhs.m_mode)
    {
        m_queue.enqueueCopyBuffer(rhs.getClBuffer(), getClBuffer(), 0, 0, rhs.getSize());
    }


    Buffer(Buffer&& rhs) noexcept = default;
    //    :cl::Buffer(std::move(rhs.getClBuffer())),
    //     m_queue(rhs.m_queue),
    //     m_mode(rhs.m_mode)
    //{
    //    rhs.m_mode = AccessMode::NotSpecified;
    //}

    Buffer& operator=(Buffer&& rhs) noexcept = default;
    Buffer& operator=(Buffer const& rhs)
    {
        cl::Buffer::operator=(rhs);
        return *this;
    }
    //{
    //    release();
    //    getClBuffer() = std::move(rhs.getClBuffer());
    //    rhs.m_mode = AccessMode::NotSpecified;
    //    return *this;
    //}

    /**
     * @brief Copy a buffer to a specified Compute Device
     */
    //TODO: fix incomplete type
    //Buffer(Buffer const& rhs, ComputeDevice& device)
    //    :cl::Buffer{device.getCLContext(), GetCLMemFlag(rhs.m_mode), rhs.getSize(), nullptr},
    //    m_queue(device.getCLQueue()),
    //    m_mode(rhs.m_mode)
    //{
    //    m_queue.enqueueCopyBuffer(rhs.getClBuffer(), getClBuffer(), 0, 0, rhs.getSize());
    //}

    Buffer& copyFrom(T const* src, size_t count, bool blocking = false)
    {
        m_queue.enqueueWriteBuffer(getClBuffer(), blocking, 0, sizeof(T) * count, src);
        return *this;
    }

    Buffer& copyTo(T* dst, size_t count, bool blocking = false)
    {
        m_queue.enqueueReadBuffer(getClBuffer(), blocking, 0, sizeof(T) * count, dst);
        return *this;
    }


};
