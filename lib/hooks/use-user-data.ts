"use client";

import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/lib/auth-context';

export function useUserData() {
  const { user } = useAuth();
  const [userData, setUserData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      setUserData(null);
      setIsLoading(false);
      return;
    }

    const fetchUserData = async () => {
      try {
        // Try to fetch user data from users table
        const { data, error } = await supabase
          .from('users')
          .select('*')
          .eq('email', user.email)
          .single();

        if (error) {
          // Only log if it's not a "no rows" error (user might not be in users table yet)
          if (error.code !== 'PGRST116') {
            console.error('Error fetching user data:', error.message || error);
          }
        } else {
          setUserData(data);
        }
      } catch (error: any) {
        if (error?.code !== 'PGRST116') {
          console.error('Error fetching user data:', error?.message || error);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [user]);

  return { userData, isLoading };
}
